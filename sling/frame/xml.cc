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

#include "sling/frame/xml.h"

#include "sling/frame/object.h"
#include "sling/frame/store.h"

namespace sling {

Frame XMLReader::Read() {
  // Parse XML input.
  if (!Parse(input_)) return Frame::nil();

  // Return frame with the top element slots.
  Slot *begin = slots_.data();
  Slot *end = slots_.data() + slots_.size();
  Handle top = store_->AllocateFrame(begin, end);
  slots_.clear();
  return Frame(store_, top);
}

bool XMLReader::StartElement(const XMLElement &element) {
  // Add empty slot with element name to stack.
  slots_.emplace_back(store_->Lookup(element.name), Handle::nil());

  // Add mark to begin new element.
  marks_.push_back(slots_.size());

  // Add slots for attributes.
  for (const XMLAttribute &attr : element.attrs) {
    Handle name = store_->Lookup(attr.name);
    Handle value = store_->AllocateString(attr.value);
    slots_.emplace_back(name, value);
  }

  return true;
}

bool XMLReader::EndElement(const char *name) {
  // Pop the begin mark for the current element.
  int begin = marks_.back();
  int end = slots_.size();
  int size = end - begin;
  marks_.pop_back();

  // Empty tags will get nil as the value.
  if (size == 0) return true;

  // If there is only one text slot for the element, the value is just the text.
  if (size == 1 && slots_.back().name.IsIs()) {
    Handle text = slots_.back().value;
    slots_.pop_back();
    slots_.back().value = text;
  } else {
    // Create frame for element.
    Slot *data = slots_.data();
    Handle frame = store_->AllocateFrame(data + begin, data + end);

    // Set frame as the element slot value.
    slots_.resize(begin);
    slots_.back().value = frame;
  }

  return true;
}

bool XMLReader::Text(const char *str) {
  // Add text slot to slot stack.
  slots_.emplace_back(Handle::is(), store_->AllocateString(str));
  return true;
}

}  // namespace sling

