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

#include "sling/frame/reader.h"

#include <string>

#include "sling/base/logging.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/string/numbers.h"

namespace sling {

Object Reader::Read() {
  return Object(store_, ReadObject());
}

Object Reader::ReadAll() {
  Handle handle = Handle::nil();
  while (!done() && !error()) {
    handle = ReadObject();
  }
  return Object(store_, handle);
}

Handle Reader::ReadObject() {
  // Read next object.
  Handle handle = ParseObject();
  if (error()) {
    stack_.reset();
    LOG(ERROR) << "Error reading object line " << line() << ":" << column()
               << ": " << error_message();
  }
  return handle;
}

Handle Reader::ParseObject() {
  Handle handle;

  switch (token()) {
    case STRING_TOKEN:
      handle = store_->AllocateString(token_text());
      NextToken();
      break;

    case INTEGER_TOKEN: {
      int32 value;
      float fvalue;
      if (safe_strto32(token_text(), &value)) {
        if (value >= Handle::kMinInt && value <= Handle::kMaxInt) {
          handle = Handle::Integer(value);
        } else {
          handle = Handle::Float(value);
        }
        NextToken();
      } else if (safe_strtof(token_text(), &fvalue)) {
        handle = Handle::Float(fvalue);
        NextToken();
      } else {
        SetError("invalid number token");
        handle = Handle::error();
      }
      break;
    }

    case FLOAT_TOKEN: {
      float value;
      if (safe_strtof(token_text(), &value)) {
        handle = Handle::Float(value);
        NextToken();
      } else {
        SetError("invalid floating point token");
        handle = Handle::error();
      }
      break;
    }

    case INDEX_TOKEN: {
      int32 value;
      if (safe_strto32(token_text(), &value)) {
        handle = Handle::Index(value);
        NextToken();
      } else {
        SetError("invalid index token");
        handle = Handle::error();
      }
      break;
    }

    case NULL_TOKEN:
      handle = Handle::nil();
      NextToken();
      break;

    case TRUE_TOKEN:
      handle = Handle::Bool(true);
      NextToken();
      break;

    case FALSE_TOKEN:
      handle = Handle::Bool(false);
      NextToken();
      break;

    case SYMBOL_TOKEN:
      handle = store_->Lookup(token_text());
      NextToken();
      break;

    case LITERAL_TOKEN:
      handle = store_->Symbol(token_text());
      NextToken();
      break;

    case NUMERIC_TOKEN:
    case INDEX_REF_TOKEN: {
      int32 index;
      if (safe_strto32(token_text(), &index)) {
        if (index >= references_.size()) references_.resize(index + 1);
        Handle &ref = Reference(index);
        if (ref.IsNil()) ref = store_->AllocateFrame(nullptr, nullptr);
        handle = ref;
        NextToken();
      } else {
        SetError("invalid numeric token");
        handle = Handle::error();
      }
      break;
    }

    case '{':
      handle = ParseFrame();
      break;

    case '[':
      handle = ParseArray();
      break;

    case ERROR:
      return Handle::error();

    default:
      SetError("syntax error");
      return Handle::error();
  }

  return handle;
}

Handle Reader::ParseFrame() {
  // Skip open bracket.
  NextToken();

  // Put frame slots on the stack while parsing.
  Word mark = Mark();

  // Parse frame slots.
  int index = -1;
  if (!json_) {
    // Parse frame in SLING format.
    while (token() != '}') {
      switch (token()) {
        case END:
          SetError("unexpected end of object");
          return Handle::error();

        case '=': {
          NextToken();
          Handle id = ParseId();
          if (error()) return Handle::error();
          if (id.IsIndex()) {
            index = id.AsIndex();
          } else {
            Push(Handle::id());
            Push(id);
          }
          break;
        }

        case ':':
          Push(Handle::isa());
          NextToken();
          Push(ParseObject());
          if (error()) return Handle::error();
          break;

        case '+':
          Push(Handle::is());
          NextToken();
          Push(ParseObject());
          if (error()) return Handle::error();
          break;

        default:
          Push(ParseObject());
          if (error()) return Handle::error();
          if (token() == ':') {
            // Slot with name and value.
            NextToken();
            Push(ParseObject());
            if (error()) return Handle::error();
          } else {
            // Slot without name.
            Handle value = Pop();
            Push(Handle::nil());
            Push(value);
          }
      }

      // Skip commas between slots.
      if (token() == ',') NextToken();
    }
  } else {
    // Parse frame in JSON format.
    while (token() != '}') {
      if (token() == END) {
        SetError("unexpected end of object");
        return Handle::error();
      }

      // Parse slot name.
      Handle name;
      if (token() == STRING_TOKEN) {
        name = store_->Lookup(token_text());
        NextToken();
      } else {
        name = ParseObject();
        if (error()) return Handle::error();
      }
      if (name.IsId()) name = store_->Lookup("_id");
      Push(name);

      // Skip colon between slot name and value.
      if (token() != ':') {
        SetError("missing colon in object slot");
        return Handle::error();
      }
      NextToken();

      // Parse slot value.
      Handle value = ParseObject();
      if (error()) return Handle::error();
      Push(value);

      // Skip commas between slots.
      if (token() == ',') NextToken();
    }
  }

  // Skip closing bracket.
  NextToken();

  // If a standby frame has been allocated for an indexed anonymous frame this
  // should be replaced by the new frame.
  Handle handle = Handle::nil();
  if (index != -1) handle = Reference(index);

  // Create new frame from slots.
  Slot *begin =  reinterpret_cast<Slot *>(stack_.address(mark));
  Slot *end =  reinterpret_cast<Slot *>(stack_.end());
  handle = store_->AllocateFrame(begin, end, handle);

  // Update reference table with new frame.
  if (index != -1) Reference(index) = handle;

  // Remove slots from stack.
  Release(mark);

  // Return handle to parsed frame.
  return handle;
}

Handle Reader::ParseArray() {
  // Skip open bracket.
  NextToken();

  // Put elements on the stack while parsing.
  Word mark = Mark();

  // Parse elements.
  while (token() != ']') {
    // Parse next element and push it on the stack.
    Push(ParseObject());
    if (error()) return Handle::error();

    // Skip commas between slots.
    if (token() == ',') NextToken();
  }

  // Skip closing bracket.
  NextToken();

  // Create new array from slots.
  Handle *begin = stack_.address(mark);
  Handle *end = stack_.end();
  Handle handle = store_->AllocateArray(begin, end);

  // Remove elements from stack.
  Release(mark);

  // Return handle to parsed array.
  return handle;
}

Handle Reader::ParseId() {
  if (token() == SYMBOL_TOKEN || token() == LITERAL_TOKEN) {
    Handle handle = store_->Symbol(token_text());
    NextToken();
    return handle;
  } else if (token() == INDEX_REF_TOKEN || token() == NUMERIC_TOKEN) {
    int32 value;
    if (safe_strto32(token_text(), &value)) {
      Handle index = Handle::Index(value);
      NextToken();
      return index;
    } else {
      SetError("invalid id");
      return Handle::error();
    }
  } else {
    return ParseObject();
  }
}

}  // namespace sling

