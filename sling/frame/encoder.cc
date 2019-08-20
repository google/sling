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

#include "sling/frame/encoder.h"

#include <string>

#include "sling/base/logging.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/frame/wire.h"
#include "sling/stream/output.h"

namespace sling {

Encoder::Encoder(const Store *store, Output *output)
    : store_(store), output_(output),
      global_(store != nullptr && store->globals() == nullptr) {
  // Insert special values in reference mapping.
  references_[Handle::nil()] = Reference(-WIRE_NIL);
  references_[Handle::id()] = Reference(-WIRE_ID);
  references_[Handle::isa()] = Reference(-WIRE_ISA);
  references_[Handle::is()] = Reference(-WIRE_IS);

  // Output binary encoding mark.
  output_->WriteChar(WIRE_BINARY_MARKER);
}

void Encoder::EncodeAll() {
  const MapDatum *map = store_->GetMap(store_->symbols());
  for (Handle *bucket = map->begin(); bucket < map->end(); ++bucket) {
    Handle h = *bucket;
    while (!h.IsNil()) {
      const SymbolDatum *symbol = store_->GetSymbol(h);
      if (symbol->bound() && !store_->IsProxy(symbol->value)) {
        EncodeObject(symbol->value);
      }
      h = symbol->next;
    }
  }
}

void Encoder::EncodeObject(Handle handle) {
  if (handle.IsRef()) {
    // Check if object has already been output.
    Reference &ref = references_[handle];
    if (ref.status == ENCODED) {
      WriteReference(ref);
    } else if (ref.status == LINKED) {
      // A link to this frame has already been encoded.
      const FrameDatum *frame = store_->GetObject(handle)->AsFrame();
      if (frame->IsProxy()) {
        WriteReference(ref);
      } else {
        // Encode a resolved frame which points back to the link reference.
        ref.status = ENCODED;
        WriteTag(WIRE_SPECIAL, WIRE_RESOLVE);
        output_->WriteVarint32(frame->slots());
        output_->WriteVarint32(ref.index);
        for (const Slot *s = frame->begin(); s < frame->end(); ++s) {
          EncodeLink(s->name);
          EncodeLink(s->value);
        }
      }
    } else {
      const Datum *datum = store_->GetObject(handle);
      switch (datum->type()) {
        case STRING: {
          // Output string contents.
          ref.index = next_index_++;
          ref.status = ENCODED;
          const StringDatum *str = datum->AsString();
          WriteTag(WIRE_STRING, str->size());
          output_->Write(str->data(), str->size());
          break;
        }

        case FRAME: {
          if (datum->IsProxy()) {
            // Output bound symbol for the proxy.
            ref.index = next_index_++;
            ref.status = LINKED;
            const ProxyDatum *proxy = datum->AsProxy();
            const SymbolDatum *symbol = store_->GetSymbol(proxy->symbol);
            EncodeSymbol(symbol, WIRE_LINK);
          } else {
            // Output frame slots.
            ref.index = next_index_++;
            ref.status = ENCODED;
            const FrameDatum *frame = datum->AsFrame();
            WriteTag(WIRE_FRAME, frame->slots());
            for (const Slot *s = frame->begin(); s < frame->end(); ++s) {
              EncodeLink(s->name);
              EncodeLink(s->value);
            }
          }
          break;
        }

        case SYMBOL:
          // Output symbol name.
          ref.index = next_index_++;
          ref.status = ENCODED;
          EncodeSymbol(datum->AsSymbol(), WIRE_SYMBOL);
          break;

        case ARRAY: {
          // Output array tag followed by array size and the elements.
          ref.index = next_index_++;
          ref.status = ENCODED;
          const ArrayDatum *array = datum->AsArray();
          WriteTag(WIRE_SPECIAL, WIRE_ARRAY);
          output_->WriteVarint32(array->length());
          for (Handle *e = array->begin(); e < array->end(); ++e) {
            EncodeLink(*e);
          }
          break;
        }

        default:
          LOG(FATAL) << "Cannot encode object handle " << handle.raw()
                     << " type " << datum->type();
      }
    }
  } else if (handle.IsInt()) {
    // Output integer as tag argument.
    WriteTag(WIRE_INTEGER, handle.AsInt());
  } else if (handle.IsIndex()) {
    // Output index using special tag.
    WriteTag(WIRE_SPECIAL, WIRE_INDEX);
    output_->WriteVarint32(handle.AsIndex());
  } else if (handle.IsFloat()) {
    // Output float as tag argument.
    WriteTag(WIRE_FLOAT, handle.FloatBits());
  } else {
    LOG(FATAL) << "Unknown handle type " << handle.raw();
  }
}

void Encoder::EncodeLink(Handle handle) {
  // Determine if only a link to the object should be output.
  Handle link = Handle::nil();
  if (handle.IsRef() && !handle.IsNil()) {
    const Datum *datum = store_->GetObject(handle);
    if (datum->IsFrame()) {
      if (datum->IsProxy()) {
        link = datum->AsProxy()->symbol;
      } else {
        const FrameDatum *frame = datum->AsFrame();
        if (frame->IsPublic()) {
          if (shallow_ || (!global_ && handle.IsGlobalRef())) {
            link = frame->get(Handle::id());
          }
        }
      }
    }
  }

  if (link.IsNil()) {
    // Encode non-reference object.
    EncodeObject(handle);
  } else {
    // Output link to object.
    Reference &ref = references_[handle];
    if (ref.status == UNRESOLVED) {
      ref.index = next_index_++;
      ref.status = LINKED;
      EncodeSymbol(store_->GetSymbol(link), WIRE_LINK);
    } else {
      WriteReference(ref);
    }
  }
}

void Encoder::EncodeSymbol(const SymbolDatum *symbol, int type) {
  const StringDatum *name = store_->GetString(symbol->name);
  WriteTag(type, name->size());
  output_->Write(name->data(), name->size());
}

void Encoder::WriteReference(const Reference &ref) {
  if (ref.index < 0) {
    // Special handles are stored with negative reference numbers.
    WriteTag(WIRE_SPECIAL, -ref.index);
  } else {
    // Output reference to previous object.
    WriteTag(WIRE_REF, ref.index);
  }
}

}  // namespace sling

