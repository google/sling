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

#include "frame/serialization.h"

#include "base/logging.h"

namespace sling {

Object FromText(Store *store, Text text) {
  StringReader reader(store, text);
  return reader.Read();
}

Object FromText(Store *store, const string &text) {
  return FromText(store, Text(text));
}

string ToText(const Store *store, Handle handle, int indent) {
  StringPrinter printer(store);
  printer.printer()->set_indent(indent);
  printer.Print(handle);
  return printer.text();
}

string ToText(const Object &object, int indent) {
  return ToText(object.store(), object.handle(), indent);
}

string ToText(const Store *store, Handle handle) {
  StringPrinter printer(store);
  printer.Print(handle);
  return printer.text();
}

string ToText(const Object &object) {
  return ToText(object.store(), object.handle());
}

Object Decode(Store *store, Text encoded) {
  StringDecoder decoder(store, encoded);
  return decoder.Decode();
}

string Encode(const Store *store, Handle handle) {
  StringEncoder encoder(store);
  encoder.Encode(handle);
  return encoder.buffer();
}

string Encode(const Object &object) {
  StringEncoder encoder(object.store());
  encoder.Encode(object.handle());
  return encoder.buffer();
}

void LoadStore(const string &filename, Store *store) {
  FileDecoder decoder(store, filename);
  store->LockGC();
  decoder.DecodeAll();
  store->UnlockGC();
}

}  // namespace sling

