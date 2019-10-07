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

#include "sling/frame/json.h"

#include <string>

#include "sling/base/logging.h"
#include "sling/frame/store.h"
#include "sling/string/numbers.h"

namespace sling {

void JSONWriter::Write(const Object &object) {
  CHECK(object.store() == nullptr ||
        object.store() == store_ ||
        object.store() == store_->globals());
  Write(object.handle());
}

void JSONWriter::Write(Handle handle) {
  if (handle.IsNil()) {
    output_->Write("null");
  } else if (handle.IsRef()) {
    const Datum *datum = store_->GetObject(handle);
    switch (datum->type()) {
      case STRING:
        WriteString(datum->AsString());
        break;

      case FRAME:
        WriteFrame(datum->AsFrame());
        break;

      case SYMBOL:
        WriteSymbol(datum->AsSymbol());
        break;

      case ARRAY:
        WriteArray(datum->AsArray());
        break;

      case INVALID:
        output_->Write("<<<invalid object>>>");
        break;

      default:
        output_->Write("<<<unknown object type>>>");
    }
  } else if (handle.IsInt()) {
    WriteInt(handle.AsInt());
  } else if (handle.IsFloat()) {
    WriteFloat(handle.AsFloat());
  } else {
    output_->Write("<<<unknown handle type>>>");
  }
}

void JSONWriter::WriteString(const StringDatum *str) {
  WriteChar('"');
  unsigned char *s = str->payload();
  unsigned char *end = str->limit();
  while (s < end) {
    switch (*s) {
      case '"': WriteChars('\\', '"'); s++; break;
      case '\\': WriteChars('\\', '\\'); s++; break;
      case '\n': WriteChars('\\', 'n'); s++; break;
      case '\t': WriteChars('\\', 't'); s++; break;
      case '\b': WriteChars('\\', 'b'); s++; break;
      case '\f': WriteChars('\\', 'f'); s++; break;
      case '\r': WriteChars('\\', 'r'); s++; break;
      default:   WriteChar(*s++);
    }
  }
  WriteChar('"');
}

void JSONWriter::WriteFrame(const FrameDatum *frame) {
  // If frame has already been output, only output a reference.
  Handle &ref = references_[frame->self];
  if (!ref.IsNil() && byref_) {
    output_->Write("{\"$ref\":");
    if (pretty()) WriteChar(' ');
    if (ref.IsIndex()) {
      WriteChars('"', '#');
      WriteInt(ref.AsIndex());
      WriteChar('"');
    } else {
      WriteLink(ref, true);
    }
    WriteChar('}');
    return;
  }

  // Increase indentation for nested frames.
  WriteChar('{');
  if (pretty()) current_indentation_ += indent_;

  // Add frame to set of printed references.
  bool first = true;
  if (frame->IsAnonymous()) {
    // Assign next local id and encode it as an index reference.
    Handle id = Handle::Index(next_index_++);
    ref = id;

    // Output index reference for anonymous frame.
    if (byref_) {
      output_->Write("\"id\":");
      if (pretty()) WriteChar(' ');
      WriteChars('"', '#');
      WriteInt(id.AsIndex());
      WriteChar('"');
      first = false;
    }
  } else {
    // Update reference table with frame id.
    ref = frame->get(Handle::id());
  }

  // Output slots.
  for (const Slot *slot = frame->begin(); slot < frame->end(); ++slot) {
    if (!first) WriteChar(',');
    if (pretty()) {
      WriteChar('\n');
      for (int i = 0; i < current_indentation_; ++i) WriteChar(' ');
    }

    WriteLink(slot->name, true);
    WriteChar(':');
    if (pretty()) WriteChar(' ');
    WriteLink(slot->value, false);

    first = false;
  }

  if (pretty()) {
    // Restore indentation.
    current_indentation_ -= indent_;
    if (frame->begin() != frame->end()) {
      WriteChar('\n');
      for (int i = 0; i < current_indentation_; ++i) WriteChar(' ');
    }
  }
  WriteChar('}');
}

void JSONWriter::WriteArray(const ArrayDatum *array) {
  WriteChar('[');
  bool first = true;
  for (Handle *element = array->begin(); element < array->end(); ++element) {
    if (!first) {
      WriteChar(',');
      if (pretty()) WriteChar(' ');
    }
    WriteLink(*element, false);
    first = false;
  }
  WriteChar(']');
}

void JSONWriter::WriteSymbol(const SymbolDatum *symbol) {
  const StringDatum *name = store_->GetString(symbol->name);
  WriteString(name);
}

void JSONWriter::WriteLink(Handle handle, bool reference) {
  // Determine if only a link to the object should be output.
  if (handle.IsRef() && !handle.IsNil()) {
    const Datum *datum = store_->GetObject(handle);
    if (datum->IsFrame()) {
      if (datum->IsProxy()) {
        // Output unresolved symbol.
        const ProxyDatum *proxy = datum->AsProxy();
        Write(proxy->symbol);
        return;
      } else {
        const FrameDatum *frame = datum->AsFrame();
        if (reference ||
            (shallow_ && frame->IsPublic()) ||
            (!global_ && handle.IsGlobalRef() && frame->IsPublic())) {
          // Output reference.
          Write(frame->get(Handle::id()));
          return;
        }
      }
    }
  }

  // Output value.
  Write(handle);
}

void JSONWriter::WriteInt(int number) {
  char buffer[kFastToBufferSize];
  char *str = FastInt32ToBuffer(number, buffer);
  output_->Write(str, strlen(str));
}

void JSONWriter::WriteFloat(float number) {
  char str[32];
  snprintf(str, 32, "%.6g", number);
  output_->Write(str, strlen(str));
}

}  // namespace sling

