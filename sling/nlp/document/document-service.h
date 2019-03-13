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

#ifndef SLING_NLP_DOCUMENT_DOCUMENT_SERVICE_H_
#define SLING_NLP_DOCUMENT_DOCUMENT_SERVICE_H_

#include "sling/frame/object.h"
#include "sling/http/web-service.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/document-tokenizer.h"
#include "sling/nlp/document/lex.h"

namespace sling {
namespace nlp {

// Conversion of SLING documents to JSON format used in document viewer.
class DocumentService {
 public:
  DocumentService(Store *commons);
  ~DocumentService();

  // Get input document.
  Document *GetInputDocument(WebService *ws) const;

  // Convert SLING document to JSON format.
  Frame Convert(const Document &document) const;

 protected:
  // Commons store.
  Store *commons_;

 private:
  // Mapping between frames and indices.
  struct FrameMapping {
    FrameMapping(Store *store) : store(store), frames(store) {
      n_name = store->Lookup("name");
    }

    // Add frame to mapping.
    bool Add(Handle handle);

    // Look up frame index for frame.
    int Lookup(Handle handle);

    // Convert value to mapped representation where frames are integer indices
    // and other values are strings.
    Handle Convert(Handle value);

    Store *store;            // frame store
    Handle n_name;           // name symbol
    Handles frames;          // frames by index
    HandleMap<int> indices;  // mapping from frame to index
  };

  // Document tokenizer.
  DocumentTokenizer tokenizer_;

  // LEX converter.
  DocumentLexer lexer_{&tokenizer_};

  // Symbol names.
  Names names_;
  DocumentNames *docnames_;
  Name n_name_{names_, "name"};
  Name n_description_{names_, "description"};
  Name n_title_{names_, "title"};
  Name n_url_{names_, "url"};
  Name n_key_{names_, "key"};
  Name n_text_{names_, "text"};
  Name n_tokens_{names_, "tokens"};
  Name n_frames_{names_, "frames"};
  Name n_types_{names_, "types"};
  Name n_slots_{names_, "slots"};
  Name n_mentions_{names_, "mentions"};
  Name n_themes_{names_, "themes"};
  Name n_evokes_{names_, "evokes"};
  Name n_simple_{names_, "simple"};
  Name n_spans_{names_, "spans"};
  Name n_begin_{names_, "begin"};
  Name n_end_{names_, "end"};
  Name n_frame_{names_, "frame"};

  Name n_item_{names_, "/w/item"};
  Name n_property_{names_, "/w/property"};
  Name n_page_item_{names_, "/wp/page/item"};
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_DOCUMENT_DOCUMENT_SERVICE_H_
