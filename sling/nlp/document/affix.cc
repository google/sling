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

#include "sling/nlp/document/affix.h"

#include <string>
#include <vector>

#include "sling/base/logging.h"
#include "sling/stream/input.h"
#include "sling/stream/output.h"
#include "sling/string/text.h"
#include "sling/util/fingerprint.h"
#include "sling/util/unicode.h"

namespace sling {
namespace nlp {

// Initial number of buckets in term and affix hash maps. This must be a power
// of two.
static const int kInitialBuckets = 1024;

static int TermHash(Text term) {
  return Fingerprint(term.data(), term.size());
}

AffixTable::AffixTable(Type type, int max_length) {
  type_ = type;
  max_length_ = max_length;
  Resize(0);
}

AffixTable::~AffixTable() { Reset(0); }

void AffixTable::Reset(int max_length) {
  // Save new maximum affix length.
  max_length_ = max_length;

  // Delete all data.
  for (size_t i = 0; i < affixes_.size(); ++i) delete affixes_[i];
  affixes_.clear();
  buckets_.clear();
  Resize(0);
}

void AffixTable::Read(InputStream *stream) {
  Input input(stream);

  // Read affix table type.
  uint32 type;
  CHECK(input.ReadVarint32(&type));
  CHECK_EQ(type_, type);

  // Read max affix length.
  uint32 max_length;
  CHECK(input.ReadVarint32(&max_length));
  Reset(max_length);

  // Read affix table size.
  uint32 size;
  CHECK(input.ReadVarint32(&size));
  Resize(size);

  // Read affixes.
  string form;
  std::vector<int> link(size, -1);
  for (int affix_id = 0; affix_id < size; ++affix_id) {
    form.clear();
    uint32 bytes, length, shorter;

    CHECK(input.ReadVarint32(&bytes));
    CHECK(input.ReadString(bytes, &form));
    CHECK(input.ReadVarint32(&length));
    if (length > 0) {
      CHECK(input.ReadVarint32(&shorter));
      link[affix_id] = shorter;
    }

    DCHECK_LE(length, max_length_);
    DCHECK(FindAffix(form) == nullptr);
    Affix *affix = AddNewAffix(form, length);
    DCHECK_EQ(affix->id(), affix_id);
  }
  DCHECK_EQ(size, affixes_.size());

  // Link affixes.
  for (int affix_id = 0; affix_id < size; ++affix_id) {
    Affix *affix = affixes_[affix_id];
    if (link[affix_id] == -1) {
      DCHECK_EQ(affix->length(), 0);
      affix->set_shorter(nullptr);
      continue;
    }

    DCHECK_GT(affix->length(), 0);
    DCHECK_GE(link[affix_id], 0);
    DCHECK_LT(link[affix_id], size);

    Affix *shorter = affixes_[link[affix_id]];
    DCHECK_EQ(affix->length(), shorter->length() + 1);
    affix->set_shorter(shorter);
  }
}

void AffixTable::Write(OutputStream *stream) const {
  Output output(stream);
  output.WriteVarint32(type_);
  output.WriteVarint32(max_length_);
  output.WriteVarint32(affixes_.size());
  for (const Affix *affix : affixes_) {
    output.WriteVarint32(affix->form().size());
    output.Write(affix->form());
    output.WriteVarint32(affix->length());
    if (affix->length() > 0) {
      CHECK(affix->shorter() != nullptr);
      output.WriteVarint32(affix->shorter()->id());
    }
  }
}

Affix *AffixTable::AddAffixesForWord(Text word) {
  // The affix length is measured in characters and not bytes so we need to
  // determine the length in characters.
  int length = UTF8::Length(word.data(), word.size());

  // Determine longest affix.
  int affix_len = length;
  if (affix_len > max_length_) affix_len = max_length_;
  if (affix_len == 0) return nullptr;

  // Find start and end of longest affix.
  const char *start;
  const char *end;
  if (type_ == PREFIX) {
    start = end = word.data();
    for (int i = 0; i < affix_len; ++i) end = UTF8::Next(end);
  } else {
    start = end = word.data() + word.size();
    for (int i = 0; i < affix_len; ++i) start = UTF8::Previous(start);
  }

  // Try to find successively shorter affixes.
  Affix *top = nullptr;
  Affix *ancestor = nullptr;
  while (affix_len >= 0) {
    // Try to find affix in table.
    Text s(start, end - start);
    Affix *affix = FindAffix(s);
    if (affix == nullptr) {
      // Affix not found, add new one to table.
      affix = AddNewAffix(s, affix_len);

      // Update ancestor chain.
      if (ancestor != nullptr) ancestor->set_shorter(affix);
      ancestor = affix;
      if (top == nullptr) top = affix;
    } else {
      // Affix found. Update ancestor if needed and return match.
      if (ancestor != nullptr) ancestor->set_shorter(affix);
      if (top == nullptr) top = affix;
      break;
    }

    // Next affix.
    if (type_ == PREFIX) {
      end = UTF8::Previous(end);
    } else {
      start = UTF8::Next(start);
    }

    affix_len--;
  }

  DCHECK(top != nullptr);
  return top;
}

Affix *AffixTable::GetAffix(int id) const {
  if (id < 0 || id >= affixes_.size()) {
    return nullptr;
  } else {
    return affixes_[id];
  }
}

string AffixTable::AffixForm(int id) const {
  Affix *affix = GetAffix(id);
  if (affix == nullptr) {
    return "";
  } else {
    return affix->form();
  }
}

int AffixTable::AffixId(Text form) const {
  Affix *affix = FindAffix(form);
  if (affix == nullptr) {
    return -1;
  } else {
    return affix->id();
  }
}

Affix *AffixTable::AddNewAffix(Text form, int length) {
  int hash = TermHash(form);
  int id = affixes_.size();
  if (id > buckets_.size()) Resize(id);
  int b = hash & (buckets_.size() - 1);

  // Create new affix object.
  Affix *affix = new Affix(id, form, length);
  affixes_.push_back(affix);

  // Insert affix in bucket chain.
  affix->next_ = buckets_[b];
  buckets_[b] = affix;

  return affix;
}

Affix *AffixTable::FindAffix(Text form) const {
  // Compute hash value for word.
  int hash = TermHash(form);

  // Try to find affix in hash table.
  Affix *affix = buckets_[hash & (buckets_.size() - 1)];
  while (affix != nullptr) {
    if (affix->form() == form) return affix;
    affix = affix->next_;
  }
  return nullptr;
}

Affix *AffixTable::GetLongestAffix(Text word) const {
  const char *start = word.data();
  const char *end = start + word.size();
  if (type_ == PREFIX) {
    const char *p = start;
    for (int i = 0; i < max_length_ && p < end; ++i) p = UTF8::Next(p);
    while (p >= start) {
      Affix *affix = FindAffix(Text(start, p - start));
      if (affix != nullptr) return affix;
      DCHECK_GT(p, start);
      p = UTF8::Previous(p);
    }
  } else {
    const char *p = end;
    for (int i = 0; i < max_length_ && p > start; ++i) p = UTF8::Previous(p);
    while (p <= end) {
      Affix *affix = FindAffix(Text(p, end - p));
      if (affix != nullptr) return affix;
      DCHECK_LT(p, end);
      p = UTF8::Next(p);
    }
  }

  LOG(FATAL) << "Should not reach here:" << word;
  return nullptr;
}

void AffixTable::Resize(int size_hint) {
  // Compute new size for bucket array.
  int new_size = kInitialBuckets;
  while (new_size < size_hint) new_size *= 2;
  int mask = new_size - 1;

  // Distribute affixes in new buckets.
  buckets_.resize(new_size);
  for (size_t i = 0; i < buckets_.size(); ++i) {
    buckets_[i] = nullptr;
  }
  for (size_t i = 0; i < affixes_.size(); ++i) {
    Affix *affix = affixes_[i];
    int b = TermHash(affix->form()) & mask;
    affix->next_ = buckets_[b];
    buckets_[b] = affix;
  }
}

}  // namespace nlp
}  // namespace sling

