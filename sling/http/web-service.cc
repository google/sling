// Copyright 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sling/http/web-service.h"

#include "sling/stream/stream.h"
#include "sling/frame/decoder.h"
#include "sling/frame/encoder.h"
#include "sling/frame/json.h"
#include "sling/frame/printer.h"
#include "sling/frame/object.h"
#include "sling/frame/reader.h"
#include "sling/frame/store.h"
#include "sling/http/http-server.h"
#include "sling/http/http-stream.h"
#include "sling/string/numbers.h"

namespace sling {

WebService::WebService(Store *commons,
                       HTTPRequest *request,
                       HTTPResponse *response)
    : store_(commons), request_(request), response_(response) {
  // Initialize input and output.
  input_ = Frame(&store_, Handle::nil());
  output_ = Frame(&store_, Handle::nil());

  // Parse URL query.
  const char *q = request->query();
  if (q != nullptr) {
    // Split query string into ampersand-separated parts.
    std::vector<Text> parts;
    const char *p = q;
    while (*q) {
      if (*q == '&') {
        parts.emplace_back(p, q - p);
        q++;
        p = q;
      } else {
        q++;
      }
    }
    parts.emplace_back(p, q - p);

    // Each part is a parameter with a name and a value.
    string name;
    string value;
    for (const Text part : parts) {
      name.clear();
      value.clear();
      int eq = part.find('=');
      if (eq != -1) {
        DecodeURLComponent(part.data(), eq, &name);
        DecodeURLComponent(part.data() + eq + 1, part.size() - eq - 1, &value);
      } else {
        DecodeURLComponent(part.data(), part.size(), &name);
      }
      parameters_.emplace_back(name, value);
    }
  }

  // Determine input format.
  Text content_type = request->content_type();
  if (content_type.empty()) {
    input_format_ = UNKNOWN;
  } else if (content_type == "application/sling") {
    input_format_ = ENCODED;
  } else if (content_type == "text/sling") {
    input_format_ = TEXT;
  } else if (content_type == "application/json") {
    input_format_ = CJSON;
  } else if (content_type == "text/json") {
    input_format_ = JSON;
  } else if (content_type == "text/lex") {
    input_format_ = LEX;
  } else if (content_type == "text/plain") {
    input_format_ = PLAIN;
  }

  // Decode input.
  if (request->content_length() > 0) {
    HTTPInputStream stream(request->buffer());
    Input in(&stream);
    switch (input_format_) {
      case ENCODED: {
        // Parse input as encoded SLING frames.
        Decoder decoder(&store_, &in);
        input_ = decoder.DecodeAll();
        break;
      }

      case TEXT:
      case COMPACT: {
        // Parse input as SLING frames in text format.
        Reader reader(&store_, &in);
        input_ = reader.Read();
        break;
      }

      case JSON:
      case CJSON: {
        // Parse input as JSON.
        Reader reader(&store_, &in);
        reader.set_json(true);
        input_ = reader.Read();
        break;
      }

      case LEX:
      case PLAIN: {
        // Get plain text from request body.
        size_t size = request->content_length();
        Handle str = store_.AllocateString(size);
        StringDatum *obj = store_.Deref(str)->AsString();
        if (in.Read(obj->data(), size)) {
          input_ = Object(&store_, str);
        }
        break;
      }

      case EMPTY:
      case UNKNOWN:
        // Ignore empty or unknown input formats.
        break;
    }
  }
}

WebService::~WebService() {
  // Do not generate a response if output is empty or if there is an error.
  if (output_.invalid()) return;
  if (response_->status() != 200) return;

  // Use input format to determine output format if it has not been set.
  if (output_format_ == EMPTY) output_format_ = input_format_;

  // Change output format based on fmt parameter.
  Text fmt = Get("fmt");
  if (!fmt.empty()) {
    if (fmt == "enc") {
      output_format_ = ENCODED;
    } else if (fmt == "txt") {
      output_format_ = TEXT;
    } else if (fmt == "lex") {
      output_format_ = LEX;
    } else if (fmt == "compact") {
      output_format_ = COMPACT;
    } else if (fmt == "json") {
      output_format_ = JSON;
    } else if (fmt == "cjson") {
      output_format_ = CJSON;
    }
  }

  // Fall back to binary encoded SLING format.
  if (output_format_ == EMPTY || output_format_ == UNKNOWN) {
    output_format_ = ENCODED;
  }

  // Output response.
  HTTPOutputStream stream(response_->buffer());
  Output out(&stream);
  switch (output_format_) {
    case ENCODED: {
      // Output as encoded SLING frames.
      response_->SetContentType("application/sling");
      Encoder encoder(&store_, &out);
      encoder.Encode(output_);
      break;
    }

    case TEXT: {
      // Output as human-readable SLING frames.
      response_->SetContentType("text/sling; charset=utf-8");
      Printer printer(&store_, &out);
      printer.set_indent(2);
      printer.set_byref(byref_);
      printer.Print(output_);
      break;
    }

    case COMPACT: {
      // Output compact SLING text.
      response_->SetContentType("text/sling; charset=utf-8");
      Printer printer(&store_, &out);
      printer.set_byref(byref_);
      printer.Print(output_);
      break;
    }

    case JSON: {
      // Output in JSON format.
      response_->SetContentType("text/json; charset=utf-8");
      JSONWriter writer(&store_, &out);
      writer.set_indent(2);
      writer.set_byref(byref_);
      writer.Write(output_);
      break;
    }

    case CJSON: {
      // Output in compact JSON format.
      response_->SetContentType("application/json; charset=utf-8");
      JSONWriter writer(&store_, &out);
      writer.set_byref(byref_);
      writer.Write(output_);
      break;
    }

    case LEX: {
      // Output is a LEX-encoded string.
      if (!output_.IsString()) {
        response_->SendError(500, "Internal Server Error", "no lex output");
      } else {
        response_->SetContentType("text/lex");
        out.Write(output_.AsString().text());
      }
      break;
    }

    case PLAIN: {
      // Output plain text string.
      if (!output_.IsString()) {
        response_->SendError(500, "Internal Server Error", "no output");
      } else {
        response_->SetContentType("text/plain");
        out.Write(output_.AsString().text());
      }
      break;
    }

    case EMPTY:
    case UNKNOWN:
      // Ignore empty or unknown output formats.
      break;
  }
}

Text WebService::Get(Text name) const {
  for (auto &p : parameters_) {
    if (p.name == name) return p.value;
  }
  return Text();
}

int WebService::Get(Text name, int defval) const {
  Text value = Get(name);
  if (value.empty()) return defval;
  int number;
  if (!safe_strto32(value.data(), value.size(), &number)) return defval;
  return number;
}

bool WebService::Get(Text name, bool defval) const {
  for (auto &p : parameters_) {
    if (p.name == name) {
      if (p.value.empty()) return true;
      if (p.value == "0") return false;
      if (p.value == "1") return true;
      if (p.value == "false") return false;
      if (p.value == "true") return true;
      if (p.value == "no") return false;
      if (p.value == "yes") return true;
      return defval;
    }
  }
  return defval;
}

}  // namespace sling

