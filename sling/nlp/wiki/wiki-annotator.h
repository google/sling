// Copyright 2018 Google Inc.
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

#ifndef SLING_NLP_WIKI_WIKI_ANNOTATOR_H_
#define SLING_NLP_WIKI_WIKI_ANNOTATOR_H_

#include "sling/frame/object.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/wiki/wiki-extractor.h"
#include "sling/string/text.h"

namespace sling {
namespace nlp {

// Abstract class for resolving Wikipedia links.
class WikiLinkResolver {
 public:
  virtual ~WikiLinkResolver() = default;

  // Resolve link to Wikipedia article returning Wikidata QID for item.
  virtual Text ResolveLink(Text link) = 0;

  // Resolve link to Wikipedia template returning Wikidata QID for item.
  virtual Text ResolveTemplate(Text link) = 0;

  // Resolve link to Wikipedia category returning Wikidata QID for item.
  virtual Text ResolveCategory(Text link) = 0;
};

// Wiki extractor sink for collecting text and annotators for Wikipedia page.
// It collects text span information about evoked frames than can later be added
// to a SLING document when the text has been tokenized. It also collects
// thematic frames for unanchored annotations.
class WikiAnnotator : public WikiTextSink {
 public:
  // Initialize document annotator. The frame annotations will be created in
  // the store and links will be resolved using the resolver.
  WikiAnnotator(Store *store, WikiLinkResolver *resolver);

  // Wiki sink interface receiving the annotations from the extractor.
  void Link(const Node &node,
            WikiExtractor *extractor,
            bool unanchored) override;
  void Template(const Node &node,
                WikiExtractor *extractor,
                bool unanchored) override;
  void Category(const Node &node,
                WikiExtractor *extractor,
                bool unanchored) override;

  // Add annotations to tokenized document.
  void AddToDocument(Document *document);

  // Add frame evoked from span.
  void AddMention(int begin, int end, Handle frame);

  // Add thematic frame.
  void AddTheme(Handle theme);

  // Add category.
  void AddCategory(Handle category);

  // Return store for annotations.
  Store *store() { return store_; }

  // Return link resolver.
  WikiLinkResolver *resolver() { return resolver_; }

 private:
  // Annotated span with byte-offset interval for the phrase in the text as well
  // as the evoked frame. The begin and end offsets are encoded as integer
  // handles to allow tracking by the frame store.
  struct Annotation {
    Annotation(int begin, int end, Handle evoked)
        : begin(Handle::Integer(begin)),
          end(Handle::Integer(end)),
          evoked(evoked) {}

    Handle begin;
    Handle end;
    Handle evoked;
  };

  // Vector of annotations that are tracked as external references.
  class Annotations : public std::vector<Annotation>, public External {
   public:
    explicit Annotations(Store *store) : External(store) {}

    void GetReferences(Range *range) override {
      range->begin = reinterpret_cast<Handle *>(data());
      range->end = reinterpret_cast<Handle *>(data() + size());
    }
  };

  // Store for frame annotations.
  Store *store_;

  // Link resolver.
  WikiLinkResolver *resolver_;

  // Annotated spans.
  Annotations annotations_;

  // Thematic frames.
  Handles themes_;

  // Categories.
  Handles categories_;

  // Symbols.
  Names names_;
  Name n_name_{names_, "name"};
  Name n_link_{names_, "/wp/link"};
  Name n_page_category_{names_, "/wp/page/category"};
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_WIKI_WIKI_ANNOTATOR_H_

