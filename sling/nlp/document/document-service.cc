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

#include "sling/nlp/document/document-service.h"

#include "sling/frame/serialization.h"

namespace sling {
namespace nlp {

DocumentService::DocumentService(Store *commons) : commons_(commons) {
  CHECK(names_.Bind(commons));
  docnames_ = new DocumentNames(commons_);
}

DocumentService::~DocumentService() {
  docnames_->Release();
}

Document *DocumentService::GetInputDocument(WebService *ws) const {
  if (ws->input().IsNil()) {
    // No input.
    return nullptr;
  } else if (ws->input().IsFrame()) {
    // Create document from frame.
    return new Document(ws->input().AsFrame(), docnames_);
  } else if (ws->input().IsString()) {
    auto format = ws->input_format();
    String input = ws->input().AsString();
    if (format == WebService::LEX) {
      // Parse LEX-encoded input document.
      Document *document = new Document(ws->store(), docnames_);
      if (!lexer_.Lex(document, input.text())) {
        delete document;
        return nullptr;
      }
      return document;
    } else if (format == WebService::PLAIN) {
      // Tokenize plain text input.
      Document *document = new Document(ws->store(), docnames_);
      tokenizer_.Tokenize(document, input.text());
      return document;
    }
  }

  // Unknown input format.
  return nullptr;
}

Frame DocumentService::Convert(const Document &document) const {
  // Builds client-side frame list.
  Store *store = document.store();
  FrameMapping mapping(store);
  Handles spans(store);
  Handles themes(store);
  mapping.Add(Handle::isa());
  mapping.Add(Handle::is());
  mapping.Add(n_name_.handle());
  mapping.Add(n_item_.handle());
  mapping.Add(n_property_.handle());

  // Add all evoked frames.
  Handles queue(store);
  for (int i = 0; i < document.num_spans(); ++i) {
    Span *span = document.span(i);
    if (span->deleted()) continue;
    const Frame &mention = span->mention();

    // Add the mention frame.
    if (mapping.Add(mention.handle())) {
      queue.push_back(mention.handle());
      int idx = mapping.Lookup(mention.handle());
      Builder mb(store);
      mb.Add(n_begin_, Handle::Integer(span->begin()));
      mb.Add(n_end_, Handle::Integer(span->end()));
      mb.Add(n_frame_, Handle::Integer(idx));
      spans.push_back(mb.Create().handle());
    }

    // Add all evoked frames.
    for (const Slot &slot : mention) {
      if (slot.name != n_evokes_) continue;

      // Queue all evoked frames.
      Handle h = slot.value;
      if (!store->IsFrame(h)) continue;
      if (mapping.Add(h)) {
        queue.push_back(h);
      }
    }
  }

  // Add thematic frames.
  for (Handle h : document.themes()) {
    if (!store->IsFrame(h)) continue;
    if (mapping.Add(h)) {
      queue.push_back(h);
    }
    int idx = mapping.Lookup(h);
    themes.push_back(Handle::Integer(idx));
  }

  // Process queue.
  int current = 0;
  while (current < queue.size()) {
    // Process all slot names and values for next frame in queue.
    Frame frame(store, queue[current++]);
    for (const Slot &slot : frame) {
      if (store->IsFrame(slot.name)) {
        if (mapping.Add(slot.name)) {
          if (slot.name.IsLocalRef()) queue.push_back(slot.name);
        }
      }
      if (store->IsFrame(slot.value)) {
        if (mapping.Add(slot.value)) {
          if (slot.value.IsLocalRef()) queue.push_back(slot.value);
        }
      }
    }
  }

  // Add basic document info.
  Builder b(store);
  b.Add(n_title_, document.top().GetHandle(n_title_));
  b.Add(n_url_, document.top().GetHandle(n_url_));
  b.Add(n_text_, document.text());
  if (document.top().Has(n_page_item_)) {
    b.Add(n_key_, document.top().GetHandle(n_page_item_));
  }
  b.Add(n_tokens_, document.top().GetHandle(n_tokens_));

  // Output frame list.
  Handles frames(store);
  Handles types(store);
  Handles slots(store);
  String idstr(store, "id");
  for (Handle handle : mapping.frames) {
    // Collect id, name, description, types, and other roles for frame.
    bool simple = false;
    Handle id = Handle::nil();
    Handle name = Handle::nil();
    Handle description = Handle::nil();
    types.clear();
    slots.clear();
    if (store->IsFrame(handle)) {
      Frame frame(store, handle);
      bool global = frame.IsGlobal();
      for (const Slot &slot : frame) {
        if (slot.name == Handle::id()) {
          if (id.IsNil()) id = slot.value;
        } else if (slot.name == n_name_) {
          if (name.IsNil()) name = slot.value;
        } else if (slot.name == n_description_ &&
                   store->IsString(slot.value)) {
          if (description.IsNil())  description = slot.value;
        } else if (slot.name.IsIsA()) {
          int idx = mapping.Lookup(slot.value);
          if (idx != -1) {
            types.push_back(Handle::Integer(idx));
          } else {
            Frame type(store, slot.value);
            if (type.valid()) {
              Handle type_id = type.id().handle();
              if (!type_id.IsNil()) types.push_back(type_id);
              if (type.GetBool(n_simple_)) simple = true;
            }
          }
        } else if (!global) {
          slots.push_back(mapping.Convert(slot.name));
          slots.push_back(mapping.Convert(slot.value));
        }
      }
    } else if (store->IsSymbol(handle)) {
      id = handle;
    }

    // Add frame to list.
    Builder fb(store);
    fb.Add(idstr, id);
    fb.Add(n_name_, name);
    fb.Add(n_description_, description);
    fb.Add(n_types_, types);
    fb.Add(n_slots_, slots);
    fb.Add(n_mentions_, Handle::nil());
    if (simple) fb.Add(n_simple_, true);
    frames.push_back(fb.Create().handle());
  }
  b.Add(n_frames_, frames);
  b.Add(n_spans_, spans);
  b.Add(n_themes_, themes);

  // Return JSON-encoded document.
  return b.Create();
}

bool DocumentService::FrameMapping::Add(Handle handle) {
  if (indices.find(handle) != indices.end()) return false;
  indices[handle] = frames.size();
  frames.push_back(handle);
  return true;
}

int DocumentService::FrameMapping::Lookup(Handle handle) {
  auto f = indices.find(handle);
  return f != indices.end() ? f->second : -1;
}

Handle DocumentService::FrameMapping::Convert(Handle value) {
  // Output null for nil values.
  if (value.IsNil()) return Handle::nil();

  if (store->IsFrame(value)) {
    // Output integer index for references to known frames.
    int frame_index = Lookup(value);
    if (frame_index != -1) return Handle::Integer(frame_index);

    // Output name or id as string for external frames.
    Handle literal = Handle::nil();
    Frame frame(store, value);
    if (frame.Has(n_name)) {
      literal = frame.GetHandle(n_name);
    } else {
      literal = frame.id().handle();
    }
    if (!literal.IsNil()) return literal;
  }

  // Output strings and symbols literally.
  if (store->IsString(value) || store->IsSymbol(value)) return value;

  // Otherwise output SLING text encoding.
  return store->AllocateString(ToText(store, value));
}

}  // namespace nlp
}  // namespace sling
