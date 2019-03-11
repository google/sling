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

// Utility classes for serializing SLING objects in text and binary format
// to and from strings and files:
//
//               String                 File
//               ------------------     ------------------
// Read text:    StringReader           FileReader
// Read binary:  StringDecoder          FileDecoder
// Write text:   StringPrinter          FilePrinter
// Write binary: StringEncoder          FileEncoder

#ifndef SLING_FRAME_SERIALIZATION_H_
#define SLING_FRAME_SERIALIZATION_H_

#include "sling/file/file.h"
#include "sling/frame/decoder.h"
#include "sling/frame/encoder.h"
#include "sling/frame/printer.h"
#include "sling/frame/reader.h"
#include "sling/stream/file.h"
#include "sling/stream/memory.h"
#include "sling/stream/stream.h"
#include "sling/string/text.h"

namespace sling {

// Read objects from string in text format.
class StringReader {
 public:
  // Initializes reading objects from a string.
  StringReader(Store *store, const char *data, size_t size)
    : stream_(data, size),
      input_(&stream_),
      reader_(store, &input_) {}
  StringReader(Store *store, Text data)
    : StringReader(store, data.data(), data.size()) {}

  // Reads next object from input.
  Object Read() { return reader_.Read(); }

  // Reads all objects from the input and returns the last value.
  Object ReadAll() { return reader_.ReadAll(); }

  // Checks if all input has been read.
  bool done() const { return reader_.done(); }

  // Returns true if errors were found while parsing input.
  bool error() const { return reader_.error(); }

  // Returns current line and column.
  int line() const { return reader_.line(); }
  int column() const { return reader_.column(); }

  // Returns last error message.
  const string &error_message() const { return reader_.error_message(); }

  // Returns underlying reader.
  Reader *reader() { return &reader_; }

 private:
  ArrayInputStream stream_;
  Input input_;
  Reader reader_;
};

// Print objects in text format to string buffer.
class StringPrinter {
 public:
  // Initializes printing objects to text.
  explicit StringPrinter(const Store *store)
    : stream_(&text_),
      output_(&stream_),
      printer_(store, &output_) {}

  // Prints object on output.
  void Print(const Object &object) { printer_.Print(object); }

  // Prints handle value relative to a store.
  void Print(Handle handle) { printer_.Print(handle); }

  // Prints all frames in the store.
  void PrintAll() { printer_.PrintAll(); }

  // Flushes the output and returns the text.
  const string &text() { output_.Flush(); return text_; }

  // Returns underlying printer.
  Printer *printer() { return &printer_; }

  // Returns underlying output.
  Output *output() { return &output_; }

 private:
  string text_;
  StringOutputStream stream_;
  Output output_;
  Printer printer_;
};

// Read binary objects from memory buffer.
class StringDecoder {
 public:
  // Initializes reading objects from a memory buffer.
  StringDecoder(Store *store, const char *data, size_t size)
    : stream_(data, size),
      input_(&stream_),
      decoder_(store, &input_) {}
  StringDecoder(Store *store, Text data)
    : StringDecoder(store, data.data(), data.size()) {}

  // Decodes the next object from the input.
  Object Decode() { return decoder_.Decode(); }

  // Decodes object from input and returns handle to it.
  Handle DecodeObject() { return decoder_.DecodeObject(); }

  // Decodes all objects from the input and returns the last value.
  Object DecodeAll() { return decoder_.DecodeAll(); }

  // Checks if all input has been read.
  bool done() { return decoder_.done(); }

  // Returns underlying decoder.
  Decoder *decoder() { return &decoder_; }

 private:
  ArrayInputStream stream_;
  Input input_;
  Decoder decoder_;
};

// Write objects in binary format to memory buffer.
class StringEncoder {
 public:
  // Initializes writing objects in binary format.
  explicit StringEncoder(const Store *store)
    : stream_(&buffer_),
      output_(&stream_),
      encoder_(store, &output_) {}

  // Encodes object to output.
  void Encode(const Object &object) { encoder_.Encode(object); }
  void Encode(Handle handle) { encoder_.Encode(handle); }

  // Encodes all frames in the store.
  void EncodeAll() { encoder_.EncodeAll(); }

  // Flushes the output and returns buffer with encoded objects.
  const string &buffer() { output_.Flush(); return buffer_; }

  // Returns underlying encoder.
  Encoder *encoder() { return &encoder_; }

 private:
  string buffer_;
  StringOutputStream stream_;
  Output output_;
  Encoder encoder_;
};

// Read objects from string in text format.
class FileReader {
 public:
  // Initializes reading objects from a file.
  FileReader(Store *store, const string &filename)
    : stream_(filename),
      input_(&stream_),
      reader_(store, &input_) {}

  // Initializes reading objects from an already opened file. This takes
  // ownership of the file.
  FileReader(Store *store, File *file)
    : stream_(file),
      input_(&stream_),
      reader_(store, &input_) {}

  // Reads next object from input.
  Object Read() { return reader_.Read(); }

  // Reads all objects from the input and returns the last value.
  Object ReadAll() { return reader_.ReadAll(); }

  // Checks if all input has been read.
  bool done() const { return reader_.done(); }

  // Returns true if errors were found while parsing input.
  bool error() const { return reader_.error(); }

  // Returns current line and column.
  int line() const { return reader_.line(); }
  int column() const { return reader_.column(); }

  // Returns last error message.
  const string &error_message() const { return reader_.error_message(); }

  // Returns underlying reader.
  Reader *reader() { return &reader_; }

 private:
  FileInputStream stream_;
  Input input_;
  Reader reader_;
};

// Write objects to file in text format.
class FilePrinter {
 public:
  // Initializes printing objects to file in text format.
  FilePrinter(const Store *store, const string &filename)
    : stream_(filename),
      output_(&stream_),
      printer_(store, &output_) {}

  // Initializes printing objects to an already opened file in text format. This
  // takes ownership of the file.
  FilePrinter(const Store *store, File *file)
    : stream_(file),
      output_(&stream_),
      printer_(store, &output_) {}

  // Flushes output and closes file.
  bool Close() { output_.Flush(); return stream_.Close(); }

  // Prints object on output.
  void Print(const Object &object) { printer_.Print(object); }

  // Prints handle value relative to a store.
  void Print(Handle handle) { printer_.Print(handle); }

  // Prints all frames in the store.
  void PrintAll() { printer_.PrintAll(); }

  // Returns underlying printer.
  Printer *printer() { return &printer_; }

 private:
  FileOutputStream stream_;
  Output output_;
  Printer printer_;
};

// Read objects from file in binary format.
class FileDecoder {
 public:
  // Initializes reading objects from a file.
  FileDecoder(Store *store, const string &filename)
    : stream_(filename),
      input_(&stream_),
      decoder_(store, &input_) {}

  // Initializes reading objects from an already opened file. This takes
  // ownership of the file.
  FileDecoder(Store *store, File *file)
    : stream_(file),
      input_(&stream_),
      decoder_(store, &input_) {}

  // Decodes the next object from the input.
  Object Decode() { return decoder_.Decode(); }

  // Decodes object from input and returns handle to it.
  Handle DecodeObject() { return decoder_.DecodeObject(); }

  // Decodes all objects from the input and returns the last value.
  Object DecodeAll() { return decoder_.DecodeAll(); }

  // Checks if all input has been read.
  bool done() { return decoder_.done(); }

  // Returns underlying decoder.
  Decoder *decoder() { return &decoder_; }

 private:
  FileInputStream stream_;
  Input input_;
  Decoder decoder_;
};

// Write objects in binary format to file.
class FileEncoder {
 public:
  // Initializes writing objects in binary format.
  FileEncoder(const Store *store, const string &filename)
    : stream_(filename),
      output_(&stream_),
      encoder_(store, &output_) {}

  // Initializes writing objects in binary format. This takes ownership of the
  // file.
  FileEncoder(const Store *store, File *file)
    : stream_(file),
      output_(&stream_),
      encoder_(store, &output_) {}

  // Flushes output and closes file.
  bool Close() { output_.Flush(); return stream_.Close(); }

  // Encodes object to output.
  void Encode(const Object &object) { encoder_.Encode(object); }
  void Encode(Handle handle) { encoder_.Encode(handle); }

  // Encodes all frames in the store.
  void EncodeAll() { encoder_.EncodeAll(); }

  // Returns underlying encoder.
  Encoder *encoder() { return &encoder_; }

 private:
  FileOutputStream stream_;
  Output output_;
  Encoder encoder_;
};

// Read objects from stream in either text or binary format.
class InputParser {
 public:
  // Initializes reading objects from a stream.
  InputParser(Store *store, InputStream *stream,
              bool force_binary = false, bool json = false);
  ~InputParser();

  // Reads next object from input.
  Object Read();

  // Reads all objects from the input and returns the last value.
  Object ReadAll();

  // Checks if all input has been read.
  bool done() const { return binary() ? decoder_->done() : reader_->done(); }

  // Returns true if errors were found while parsing input.
  bool error() const { return !binary() && reader_->error(); }

  // Returns current line and column.
  int line() const { return binary() ? 0 : reader_->line(); }
  int column() const { return binary() ? 0 : reader_->column(); }

  // Returns last error message.
  const string &error_message() const {
    static const string empty;
    return binary() ? empty : reader_->error_message();
  }

  // Checks if input is binary encoded.
  bool binary() const { return decoder_ != nullptr; }

 private:
  Input input_;
  Decoder *decoder_ = nullptr;
  Reader *reader_ = nullptr;
};

// Reads object in text format from string.
Object FromText(Store *store, Text text);
Object FromText(Store *store, const string &text);

// Returns string with object in text format.
string ToText(const Store *store, Handle handle, int indent);
string ToText(const Object &object, int indent);
string ToText(const Store *store, Handle handle);
string ToText(const Object &object);

// Decodes object from string buffer.
Object Decode(Store *store, Text encoded);

// Encodes object into string buffer.
string Encode(const Store *store, Handle handle);
string Encode(const Object &object);

// Load store from file.
void LoadStore(const string &filename, Store *store);

}  // namespace sling

#endif  // SLING_FRAME_SERIALIZATION_H_

