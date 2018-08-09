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

#ifndef SLING_NLP_DOCUMENT_AFFIX_H_
#define SLING_NLP_DOCUMENT_AFFIX_H_

#include <string>
#include <vector>

#include "sling/base/macros.h"
#include "sling/base/types.h"
#include "sling/stream/stream.h"
#include "sling/string/text.h"

namespace sling {
namespace nlp {

// An affix represents a prefix or suffix of a word of a certain length. Each
// affix has a unique id and a textual form. An affix also has a pointer to the
// affix that is one character shorter. This creates a chain of affixes that are
// successively shorter.
class Affix {
 private:
  friend class AffixTable;
  Affix(int id, Text form, int length)
      : id_(id),
        length_(length),
        form_(form.data(), form.size()),
        shorter_(nullptr),
        next_(nullptr) {}

 public:
  // Returns unique id of affix.
  int id() const { return id_; }

  // Returns the textual representation of the affix.
  const string &form() const { return form_; }

  // Returns the length of the affix.
  int length() const { return length_; }

  // Gets/sets the affix that is one character shorter.
  Affix *shorter() const { return shorter_; }
  void set_shorter(Affix *next) { shorter_ = next; }

 private:
  // Affix id.
  int id_;

  // Length (in characters) of affix.
  int length_;

  // Text form of affix.
  string form_;

  // Pointer to affix that is one character shorter.
  Affix *shorter_;

  // Next affix in bucket chain.
  Affix *next_;

  DISALLOW_COPY_AND_ASSIGN(Affix);
};

// An affix table holds all prefixes/suffixes of all the words added to the
// table up to a maximum length. The affixes are chained together to enable
// fast lookup of all affixes for a word.
class AffixTable {
 public:
  // Affix table type.
  enum Type { PREFIX = 0, SUFFIX = 1 };

  AffixTable(Type type, int max_length);
  ~AffixTable();

  // Resets the affix table and initialize the table for affixes of up to the
  // maximum length specified.
  void Reset(int max_length);

  // Read affix table from input stream.
  void Read(InputStream *stream);

  // Write affix table to output stream.
  void Write(OutputStream *stream) const;

  // Adds all prefixes/suffixes of the word up to the maximum length to the
  // table. The longest affix is returned. The pointers in the affix can be
  // used for getting shorter affixes.
  Affix *AddAffixesForWord(Text word);

  // Gets the affix information for the affix with a certain id. Returns null if
  // there is no affix in the table with this id.
  Affix *GetAffix(int id) const;

  // Finds the longest affix for word.
  Affix *GetLongestAffix(Text word) const;

  // Gets affix form from id. If the affix does not exist in the table, an empty
  // string is returned.
  string AffixForm(int id) const;

  // Gets affix id for affix. If the affix does not exist in the table, -1 is
  // returned.
  int AffixId(Text form) const;

  // Returns size of the affix table.
  int size() const { return affixes_.size(); }

  // Returns the maximum affix length.
  int max_length() const { return max_length_; }

 private:
  // Adds a new affix to table.
  Affix *AddNewAffix(Text form, int length);

  // Finds existing affix in table.
  Affix *FindAffix(Text form) const;

  // Resizes bucket array.
  void Resize(int size_hint);

  // Affix type (prefix or suffix).
  Type type_;

  // Maximum length of affix.
  int max_length_;

  // Index from affix ids to affix items.
  std::vector<Affix *> affixes_;

  // Buckets for word-to-affix hash map.
  std::vector<Affix *> buckets_;

  DISALLOW_COPY_AND_ASSIGN(AffixTable);
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_DOCUMENT_AFFIX_H_

