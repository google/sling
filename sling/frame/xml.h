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

#ifndef SLING_FRAME_XML_H_
#define SLING_FRAME_XML_H_

#include <vector>

#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/stream/input.h"
#include "sling/web/xml-parser.h"

namespace sling {

// The XML reader parses XML input and converts it into frame format. Each XML
// element is converted into a frame slot where the name is the XML tag name and
// the value is the content of the XML element. XML attributes and child
// elements are converted into slots in a sub-frame. If the XML element only
// contains text, the value is just a string with the text.
class XMLReader : public XMLParser {
 public:
  // Initializes XML reader with store and input.
  XMLReader(Store *store, Input *input)
      : store_(store), input_(input), slots_(store) {}

  // Parse XML input and return frame with content or nil on errors.
  Frame Read();

 private:
  // Callbacks from XML parser.
  bool StartElement(const XMLElement &element) override;
  bool EndElement(const char *name) override;
  bool Text(const char *str) override;

  // Object store.
  Store *store_;

  // Input with XML.
  Input *input_;

  // Stack with slots for the elements currently being parsed.
  Slots slots_;

  // Stack which marks the first slot for the elements being parsed.
  std::vector<int> marks_;
};

}  // namespace sling

#endif  // SLING_FRAME_XML_H_

