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

#include <string.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/nlp/wiki/wiki.h"
#include "sling/stream/file-input.h"
#include "sling/string/numbers.h"
#include "sling/string/printf.h"
#include "sling/task/frames.h"
#include "sling/task/process.h"
#include "sling/task/task.h"
#include "sling/web/xml-parser.h"

namespace sling {
namespace nlp {

// Parser for parsing Wikipedia XML dump.
class WikipediaXMLParser : public XMLParser {
 public:
  WikipediaXMLParser(task::Task *task) : task_(task) {
    // Get output channel for articles and redirects.
    articles_channel_ = task->GetSink("articles");
    redirects_channel_ = task->GetSink("redirects");
    categories_channel_ = task->GetSink("categories");

    // Set up XML field mapping.
    fields_.resize(NUM_FIELDS);
    element_map_["mediawiki"] = MEDIAWIKI;
    element_map_["page"] = PAGE;
    element_map_["id"] = ID;
    element_map_["ns"] = NS;
    element_map_["redirect"] = REDIRECT;
    element_map_["revision"] = REVISION;
    element_map_["title"] = TITLE;
    element_map_["text"] = TEXT;

    // Initialize commons store.
    names_.Bind(&commons_);
    commons_.Freeze();

    // Initialize counters.
    num_articles_ = task->GetCounter("wikipedia_articles");
    num_categories_ = task->GetCounter("wikipedia_categories");
    num_redirects_ = task->GetCounter("wikipedia_redirects");
    num_fragment_redirects_ = task->GetCounter("fragment_redirects");
    input_bytes_ = task->GetCounter("wikipedia_input_bytes");
  }

  ~WikipediaXMLParser() {
    if (articles_channel_) articles_channel_->Close();
    if (redirects_channel_) redirects_channel_->Close();
  }

  // Handle XML start element.
  bool StartElement(const XMLElement &element) override {
    // Lookup element.
    auto f = element_map_.find(element.name);
    if (f == element_map_.end()) {
      field_ = NONE;
      return true;
    }

    switch (f->second) {
      case PAGE:
        fields_[ID].clear();
        fields_[NS].clear();
        fields_[REDIRECT].clear();
        fields_[TITLE].clear();
        fields_[TEXT].clear();
        redirect_.clear();
        break;

      case REDIRECT:
        redirect_ = element.Get("title", "");
        break;

      case REVISION:
        in_revision_ = true;
        break;

      case MEDIAWIKI:
        lang_ = element.Get("xml:lang", "en");
        if (lang_ == "nb") lang_ = "no";
        break;

      default:
        // Set current field.
        field_ = f->second;
        if (in_revision_ && field_ == ID) field_ = REVISIONID;
        fields_[field_].clear();
    }

    return true;
  }

  // Handle XML end element.
  bool EndElement(const char *name) override {
    // Lookup element.
    auto f = element_map_.find(name);
    if (f == element_map_.end()) {
      field_ = NONE;
      return true;
    }

    switch (f->second) {
      case PAGE:
        ProcessPage();
        break;

      case REVISION:
        in_revision_ = false;
        break;

      default:
        field_ = NONE;
    }

    return true;
  }

  // Handle XML text.
  bool Text(const char *str) override {
    // Append text to current field.
    if (field_ != NONE) fields_[field_].append(str);
    return true;
  }

  // Process Wikipedia page.
  void ProcessPage() {
    // Get article fields.
    const string &title = fields_[TITLE];
    const string &text = fields_[TEXT];
    int ns = -1;
    if (!fields_[NS].empty()) {
      CHECK(safe_strto32(fields_[NS], &ns));
    }
    int pageid = -1;
    if (!fields_[ID].empty()) {
      CHECK(safe_strto32(fields_[ID], &pageid));
    }
    string id = Wiki::Id(lang_, title);

    if (redirect_.empty()) {
      // Only keep articles in main and category namespaces.
      task::Counter *&ctr = num_namespace_pages_[ns];
      if (ctr == nullptr) {
        ctr = task_->GetCounter(StringPrintf("namespace_pages[%d]", ns));
      }
      ctr->Increment();
      if (ns != WIKIPEDIA_NAMESPACE_MAIN &&
          ns != WIKIPEDIA_NAMESPACE_CATEGORY) {
        return;
      }

      // Build article frame.
      Store store(&commons_);
      Builder builder(&store);
      builder.AddId(id);
      builder.AddIsA(n_page_);
      if (ns == WIKIPEDIA_NAMESPACE_CATEGORY) builder.AddIsA(n_category_);
      builder.Add(n_page_pageid_, pageid);
      builder.Add(n_page_title_, title);
      builder.AddLink(n_lang_, "/lang/" + lang_);
      if (!text.empty()) {
        builder.Add(n_page_text_, text);
      }

      // Output frame.
      Frame frame = builder.Create();
      if (ns == WIKIPEDIA_NAMESPACE_MAIN) {
        if (articles_channel_) {
          articles_channel_->Send(task::CreateMessage(frame));
        }
        num_articles_->Increment();
      } else if (ns == WIKIPEDIA_NAMESPACE_CATEGORY) {
        if (categories_channel_) {
          categories_channel_->Send(task::CreateMessage(frame));
        }
        num_categories_->Increment();
      }
    } else {
      // Ignore redirects with fragments.
      if (fields_[TEXT].find('#', 1) != string::npos) {
        VLOG(9) << "Ignore redirect from " << title << ": " << fields_[TEXT];
        num_fragment_redirects_->Increment();
      } else {
        // Build redirect frame.
        Store store(&commons_);
        Builder builder(&store);
        builder.AddId(id);
        builder.AddIsA(n_redirect_);
        builder.Add(n_redirect_pageid_, pageid);
        builder.Add(n_redirect_title_, title);
        if (!redirect_.empty()) {
          builder.AddLink(n_redirect_link_, Wiki::Id(lang_, redirect_));
        }

        // Output frame on redirect channel.
        if (redirects_channel_) {
          Frame frame = builder.Create();
          redirects_channel_->Send(task::CreateMessage(frame));
        }
        num_redirects_->Increment();
      }
    }

    // Update input statistics.
    uint64 bytes = input()->stream()->ByteCount();
    input_bytes_->Increment(bytes - position_);
    position_ = bytes;
  }

 private:
  // Wikimedia XML fields.
  enum Field {
    NONE, MEDIAWIKI, PAGE, REVISION,
    ID, REVISIONID, NS, REDIRECT, TITLE, TEXT,
    NUM_FIELDS
  };

  // Task.
  task::Task *task_;

  // Output channels.
  task::Channel *articles_channel_;
  task::Channel *redirects_channel_;
  task::Channel *categories_channel_;

  // Wikipedia language.
  string lang_;

  // XML field values.
  std::vector<string> fields_;

  // Wikipedia redirect.
  string redirect_;

  // Field mapping.
  std::unordered_map<string, Field> element_map_;

  // Current field.
  Field field_ = NONE;
  bool in_revision_ = false;

  // Statistics.
  std::unordered_map<int, task::Counter *> num_namespace_pages_;
  task::Counter *num_articles_;
  task::Counter *num_redirects_;
  task::Counter *num_categories_;
  task::Counter *num_fragment_redirects_;
  task::Counter *input_bytes_;
  uint64 position_ = 0;

  // SLING store.
  Store commons_;
  Names names_;
  Name n_lang_{names_, "lang"};

  Name n_page_{names_, "/wp/page"};
  Name n_page_pageid_{names_, "/wp/page/pageid"};
  Name n_page_title_{names_, "/wp/page/title"};
  Name n_page_text_{names_, "/wp/page/text"};
  Name n_category_{names_, "/wp/category"};

  Name n_redirect_{names_, "/wp/redirect"};
  Name n_redirect_pageid_{names_, "/wp/redirect/pageid"};
  Name n_redirect_title_{names_, "/wp/redirect/title"};
  Name n_redirect_link_{names_, "/wp/redirect/link"};
};

// Parse Wikipedia dump and convert to SLING documents.
class WikipediaImporter : public task::Process {
 public:
  // Process input file.
  void Run(task::Task *task) override {
    // Get input file.
    task::Binding *input = task->GetInput("input");
    CHECK(input != nullptr) << "No input resource";

    // Open input file.
    int buffer_size = task->Get("buffer_size", 256 * 1024);
    FileInput file(input->resource()->name(), buffer_size);

    // Parse XML parser.
    WikipediaXMLParser parser(task);
    CHECK(parser.Parse(&file));
  }
};

REGISTER_TASK_PROCESSOR("wikipedia-importer", WikipediaImporter);

}  // namespace nlp
}  // namespace sling

