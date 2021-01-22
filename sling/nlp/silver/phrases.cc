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

#include <string>
#include <vector>

#include "sling/frame/serialization.h"
#include "sling/nlp/document/annotator.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/lex.h"
#include "sling/nlp/kb/facts.h"
#include "sling/nlp/kb/phrase-table.h"
#include "sling/nlp/silver/chart.h"
#include "sling/stream/file-input.h"
#include "sling/util/fingerprint.h"
#include "sling/util/mutex.h"

namespace sling {
namespace nlp {

using namespace task;

// Annotate resolved mentions with internal structure using the knowledge base
// and alias table to identify sub-mentions that are related to the frame(s)
// evoked by the mention.
class PhraseStructureAnnotator : public Annotator {
 public:
  void Init(Task *task, Store *commons) override {
    // Load phrase table.
    string alias_file = task->GetInputFile("aliases");
    aliases_ = PhraseTable::Acquire(task, commons, alias_file);

    // Initialize fact extractor.
    catalog_.Init(commons);

    // Initialize phrase cache.
    cache_size_ = task->Get("phrase_cache_size", 1024 * 1024);
    cache_.resize(cache_size_);
    for (const string &filename : task->GetInputFiles("phrases")) {
      LoadCache(filename);
    }

    // Forbid some properties that are rarely expressed in phrases.
    static const char *forbidden[] = {
      "P22",    // father
      "P25",    // mother
      "P40",    // child
      "P47",    // shares border with
      "P530",   // diplomatic relation
      "P190",   // twinned administrative body
      "P1336",  // territory claimed by
      "P1365",  // replaces
      "P1366",  // replaced by
      "P155",   // follows
      "P460",   // said to be the same as
      nullptr
    };
    for (const char **ptr = forbidden; *ptr; ++ptr) {
      Handle property = commons->LookupExisting(*ptr);
      if (property.IsNil()) {
        LOG(WARNING) << "Unknown property" << *ptr;
        continue;
      }
      forbidden_.insert(property);
    }
  }

  // Annotate multi-word expressions in document with phrase structures.
  void Annotate(Document *document) override {
    // Find all resolved multi-word expressions.
    Store *store = document->store();
    for (Span *span : document->spans()) {
      if (span->length() < 2) continue;
      Handle frame = span->evoked();
      if (frame.IsNil()) continue;

      // Get resolved item id for evoked frame.
      Handle entity = store->Resolve(frame);
      string id = store->FrameId(entity).str();
      if (id.empty()) continue;

      // Look up phrase in cache.
      string annotations;
      if (LookupPhrase(id, span->GetText(), span->length(), &annotations)) {
        // Add cached phrase annotations.
        if (!annotations.empty()) {
          // Decode cached phrase annotations.
          Frame top = Decode(store, annotations).AsFrame();
          Document phrase(top, document->names());

          // Add phrase annotations to document.
          Merge(document, phrase, span->begin());
        }
      } else {
        // Get sub document with phrase span.
        Document phrase(*document, span->begin(), span->end(), false);

        // Analyze phrase structure of span.
        if (AnalyzePhrase(id, &phrase) != Handle::nil()) {
          // Add phrase annotations to document.
          Merge(document, phrase, span->begin());
        }
      }
    }
  }

  // Analyze phrase structure and return frame evoke from phrase.
  Handle AnalyzePhrase(const string &id, Document *phrase) {
    // Get facts for entity.
    Store *store = phrase->store();
    Handle item = store->LookupExisting(id);
    if (item.IsNil()) return Handle::nil();
    CHECK(item.IsGlobalRef());
    Facts facts(&catalog_);
    facts.Extract(item);
    HandleSet targets;
    for (int i = 0; i < facts.size(); ++i) {
      if (!facts.simple(i)) continue;
      if (forbidden_.count(facts.first(i)) > 0) continue;
      targets.insert(facts.last(i));
    }

    // Try to match all subphrases to entities in the target set.
    int length = phrase->length();
    SpanChart chart(phrase, 0, length, length);
    Handles matches(store);
    bool matches_found = false;
    for (int b = 0; b < length; ++b) {
      if (phrase->token(b).skipped()) continue;
      for (int e = b + 1; e <= (b == 0 ? length - 1 : length); ++e) {
        if (phrase->token(e - 1).skipped()) continue;

        // Look up subphrase in phrase table.
        uint64 fp = phrase->PhraseFingerprint(b, e);
        SpanChart::Item &span = chart.item(b, e);
        span.matches = aliases_->Find(fp);
        if (span.matches == nullptr) continue;

        // Check if any target can match the subphrase.
        aliases_->GetMatches(span.matches, &matches);
        for (Handle h : matches) {
          if (targets.count(h) > 0) {
            // Match found.
            span.aux = h;
            break;
          }
        }

        // Set the span cost to one if there are any matches.
        if (!span.aux.IsNil()) {
          span.cost = 1.0;
          matches_found = true;
        } else {
          span.matches = nullptr;
        }
      }
    }

    // Check if any matching subphrases were found.
    if (!matches_found) {
      // Update cache with negative result.
      CachePhrase(id, phrase->text(), phrase->length(), "");
      return Handle::nil();
    }

    // Compute best span covering.
    chart.Solve();

    // Build frame for phrase.
    Builder frame(store);
    frame.AddIs(item);

    // Analyze all matched subphrases.
    chart.Extract([&](int begin, int end, const SpanChart::Item &span) {
      // Determine relation between entities for phrase and subphrase.
      Handle target = span.aux;
      CHECK(!target.IsNil());
      Handle relation = Handle::nil();
      for (int i = 0; i < facts.size(); ++i) {
        if (facts.last(i) == target && facts.simple(i)) {
          Handle property = facts.first(i);
          if (forbidden_.count(property) == 0) {
            relation = facts.first(i);
            break;
          }
        }
      }
      if (relation.IsNil()) return;

      // A subphrase cannot resolve to the same meaning as the whole phrase.
      if (target == item) return;

      // Look up subphrase in cache.
      string subid = store->FrameId(target).str();
      CHECK(!subid.empty());
      string annotations;
      Handle subevoke = Handle::nil();
      string subphrase = phrase->PhraseText(begin, end);
      if (LookupPhrase(subid, subphrase, end - begin, &annotations)) {
        if (!annotations.empty()) {
          // Add cached phrase annotations.
          Frame top = Decode(store, annotations).AsFrame();
          Document subphrase(top, phrase->names());
          Merge(phrase, subphrase, begin);
          Span *subspan = phrase->GetSpan(begin, end);
          if (subspan != nullptr) subevoke = subspan->evoked();
        }
      } else {
        // Subphrase not found in cache. Get document for subphrase.
        Document subphrase(*phrase, begin, end, false);

        // Recursively analyze phrase structure of subphrase.
        subevoke = AnalyzePhrase(subid, &subphrase);
        if (!subevoke.IsNil()) {
          // Add phrase annotations to document.
          Merge(phrase, subphrase, begin);
        }
      }

      // Add simple frame for subphrase if no structure was found.
      if (subevoke.IsNil()) {
        Frame subframe = Builder(store).AddIs(subid).Create();
        phrase->AddSpan(begin, end)->Evoke(subframe);
        subevoke = subframe.handle();
      }

      // Add relation between entities for phrase and subphrase.
      frame.Add(relation, subevoke);
    });

    // Evoke frame for whole phrase.
    phrase->AddSpan(0, length)->Evoke(frame.Create());

    // Add phrase annotations to cache.
    phrase->Update();
    CachePhrase(id, phrase->text(), phrase->length(), Encode(phrase->top()));

    return frame.handle();
  }

  // Look up phrase in phrase annotation cache. Return true if the phrase is
  // found.
  bool LookupPhrase(const string &id, const string &text, int len,
                    string *annotations) {
    MutexLock lock(&mu_);
    Phrase &phrase = cache_[Hash(id, text, len) % cache_size_];
    if (id != phrase.id) return false;
    if (text != phrase.text) return false;
    if (len != phrase.len) return false;
    *annotations = phrase.annotations;
    return true;
  }

  // Add phrase annotations for entity alias to cache.
  bool CachePhrase(const string &id, const string &text, int len,
                   const string &annotations, bool sticky = false) {
    MutexLock lock(&mu_);
    Phrase &phrase = cache_[Hash(id, text, len) % cache_size_];
    if (phrase.sticky) {
      // Never overwrite stick annotations.
      if (sticky) {
        LOG(WARNING) << "Sticky phrase collsion for " << id << ": '"
                     << phrase.text << "' and '" << text;
      }
      return false;
    }
    phrase.id = id;
    phrase.text = text;
    phrase.len = len;
    phrase.annotations = annotations;
    phrase.sticky = sticky;
    return true;
  }

  // Load custom phrase annotations into cache.
  void LoadCache(const string &filename) {
    // Initialize document store for read phrase annotations.
    Store store;
    DocumentNames *names = new DocumentNames(&store);
    DocumentTokenizer tokenizer;
    DocumentLexer lexer(&tokenizer);

    // Read phrase annotations from file.
    FileInput input(filename);
    string line;
    while (input.ReadLine(&line)) {
      // Skip blank lines and comments.
      while (!line.empty()) {
        char ch = line.back();
        if (ch != ' ' && ch != '\n') break;
        line.pop_back();
      }
      if (line.empty() || line[0] == ';') continue;

      // Read LEX-encoded phrase annotations.
      Document phrase(&store, names);
      CHECK(lexer.Lex(&phrase, line)) << line;

      // Get item id for phrase.
      Span *span = phrase.GetSpan(0, phrase.length());
      CHECK(span != nullptr) << line;
      Text id = store.FrameId(store.Resolve(span->evoked()));
      CHECK(!id.empty()) << line;

      // Add sticky phrase annotations to cache.
      CachePhrase(id.str(), phrase.text(), phrase.length(),
                  Encode(phrase.top()), true);
    }
    names->Release();
  }

  // Compute hash for id and phrase text.
  static uint32 Hash(const string &id, const string &text, int len) {
    uint32 fp1 = Fingerprint32(id.data(), id.size());
    uint32 fp2 = Fingerprint32(text.data(), text.size());
    return fp1 ^ fp2 ^ len;
  }

  // Merge annotations for phrase into document at position.
  static void Merge(Document *document, const Document &phrase, int pos) {
    int length = phrase.length();
    CHECK_GE(document->length(), pos + length);
    for (Span *span : phrase.spans()) {
      // Add new span to document (or get an existing span).
      Span *docspan = document->AddSpan(span->begin() + pos, span->end() + pos);

      // Get frame evoked from phrase span.
      Frame evoked = span->Evoked();
      if (evoked.IsNil()) continue;

      // Import or merge evoked frame from phrase into document.
      Frame existing = docspan->Evoked();
      if (existing.IsNil()) {
        // Import evoked frame from phrase.
        docspan->Evoke(evoked);
      } else if (existing.IsPublic()) {
        // Replace existing frame.
        docspan->Replace(existing, evoked);
      } else if (evoked.IsPublic()) {
        // Add is: slot with evoked frame to existing frame.
        if (!existing.Is(evoked)) existing.AddIs(evoked);
      } else {
        // Merge existing frame with phrase frame.
        Builder b(existing);
        for (const Slot &s : evoked) {
          if (s.name == Handle::is() && existing.Is(s.value)) continue;
          b.Add(s.name, s.value);
        }
        b.Update();
      }
    }
  }

 private:
  // Cached phrase with name structure annotations.
  struct Phrase {
    string id;             // entity id for phrase name
    string text;           // phrase text
    int len;               // number of tokens in phrase
    string annotations;    // phrase annotations as encoded SLING frames
    bool sticky;           // custom annotations are sticky
  };

  // Phrase table with aliases.
  const PhraseTable *aliases_ = nullptr;

  // Fact catalog for fact extraction.
  FactCatalog catalog_;

  // Phrase annotation cache.
  std::vector<Phrase> cache_;
  uint32 cache_size_;

  // Forbidden relation types.
  HandleSet forbidden_;

  // Mutex for accessing cache.
  Mutex mu_;
};

REGISTER_ANNOTATOR("phrase-structure", PhraseStructureAnnotator);

}  // namespace nlp
}  // namespace sling

