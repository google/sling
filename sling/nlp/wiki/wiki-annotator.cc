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

#include "sling/nlp/wiki/wiki-annotator.h"

namespace sling {
namespace nlp {

WikiAnnotator::WikiAnnotator(Store *store, WikiLinkResolver *resolver)
    : store_(store),
      resolver_(resolver),
      annotations_(store),
      themes_(store),
      categories_(store) {
  names_.Bind(store);
}

void WikiAnnotator::Link(const Node &node,
                         WikiExtractor *extractor,
                         bool unanchored) {
  // Resolve link.
  Text link = resolver_->ResolveLink(node.name());
  if (link.empty()) return;

  if (unanchored) {
    // Extract anchor as plain text.
    WikiPlainTextSink plain;
    extractor->Enter(&plain);
    extractor->ExtractChildren(node);
    extractor->Leave(&plain);

    // Add thematic frame for link.
    if (!plain.text().empty()) {
      Builder theme(store_);
      theme.AddIsA(n_link_);
      theme.Add(n_name_, plain.text());
      theme.AddIs(store_->Lookup(link));
      AddTheme(theme.Create().handle());
    }
  } else {
    // Output anchor text.
    int begin = position();
    extractor->ExtractChildren(node);
    int end = position();

    // Evoke frame for link.
    if (begin != end) {
      AddMention(begin, end, store_->Lookup(link));
    }
  }
}

void WikiAnnotator::Template(const Node &node,
                             WikiExtractor *extractor,
                             bool unanchored) {
  extractor->ExtractSkip(node);
}

void WikiAnnotator::Category(const Node &node,
                             WikiExtractor *extractor,
                             bool unanchored) {
  // Resolve link.
  Text link = resolver_->ResolveCategory(node.name());
  if (link.empty()) return;

  // Add category link.
  AddCategory(store_->Lookup(link));
}

void WikiAnnotator::AddToDocument(Document *document) {
  // Add annotated spans to document.
  for (Annotation &a : annotations_) {
    int begin = document->Locate(a.begin.AsInt());
    int end = document->Locate(a.end.AsInt());
    Span *span = document->AddSpan(begin, end);
    span->Evoke(a.evoked);
  }

  // Add thematic frames.
  for (Handle theme : themes_) {
    document->AddTheme(theme);
  }

  // Add categories.
  for (Handle category : categories_) {
    document->AddExtra(n_page_category_, category);
  }
}

void WikiAnnotator::AddMention(int begin, int end, Handle frame) {
  annotations_.emplace_back(begin, end, frame);
}

void WikiAnnotator::AddTheme(Handle theme) {
  themes_.push_back(theme);
}

void WikiAnnotator::AddCategory(Handle category) {
  categories_.push_back(category);
}

}  // namespace nlp
}  // namespace sling

