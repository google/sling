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
#include <unordered_map>

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/document-tokenizer.h"
#include "sling/nlp/wiki/wikipedia-map.h"
#include "sling/nlp/wiki/wiki-annotator.h"
#include "sling/nlp/wiki/wiki-extractor.h"
#include "sling/nlp/wiki/wiki-parser.h"
#include "sling/task/frames.h"
#include "sling/task/reducer.h"
#include "sling/task/task.h"
#include "sling/string/numbers.h"
#include "sling/string/strcat.h"
#include "sling/string/text.h"

namespace sling {
namespace nlp {

static bool strtoint(Text text, int *value) {
  return safe_strto32(text.data(), text.size(), value);
}

// Convert Wikipedia articles to documents. This parses the wikitext in the
// articles and extracts the plain text. Anchors are converted to mentions
// of the linked entity, and all links are converted to Wikidata ids, including
// resolution of redirects. Titles, anchors, and disambiguations are used for
// building up the aliases table for each entity.
class WikipediaDocumentBuilder : public task::FrameProcessor,
                                 public WikiLinkResolver {
 public:
  void Startup(task::Task *task) override {
    // Get language settings.
    language_ = task->Get("language", "en");
    lang_ = commons_->Lookup(StrCat("/lang/" + language_));
    Frame langinfo(commons_, lang_);
    CHECK(langinfo.valid());
    category_prefix_ = langinfo.GetString(n_lang_category_);
    template_prefix_ = langinfo.GetString(n_lang_template_);
    task->Fetch("skip_tables", &skip_tables_);

    // Load redirects.
    task::Binding *redir = CHECK_NOTNULL(task->GetInput("redirects"));
    LOG(INFO) << "Loading redirects from " << redir->resource()->name();
    wikimap_.LoadRedirects(redir->resource()->name());

    // Load id mapping.
    task::Binding *wikimap = CHECK_NOTNULL(task->GetInput("wikimap"));
    LOG(INFO) << "Loading wikimap from " << wikimap->resource()->name();
    wikimap_.LoadMapping(wikimap->resource()->name());

    wikimap_.Freeze();
    task->GetCounter("wiki_redirects")->Increment(wikimap_.redirects().size());

    // Initialize document schema.
    docnames_ = new DocumentNames(commons_);

    // Get counters.
    num_article_pages_ = task->GetCounter("article_pages");
    num_category_pages_ = task->GetCounter("category_pages");
    num_disambiguation_pages_ = task->GetCounter("disambiguation_pages");
    num_list_pages_ = task->GetCounter("list_pages");
    num_unknown_pages_ = task->GetCounter("unknown_wikipedia_pages");
    num_unexpected_pages_ = task->GetCounter("unexpected_pages");
    num_wiki_ast_nodes_ = task->GetCounter("wiki_ast_nodes");
    num_article_text_bytes_ = task->GetCounter("article_text_bytes");
    num_article_tokens_ = task->GetCounter("article_tokens");
    num_links_ = task->GetCounter("wiki_links");
    num_dead_links_ = task->GetCounter("dead_links");
    num_fragment_links_ = task->GetCounter("fragment_links");
    num_categories_ = task->GetCounter("wiki_categories");
    num_unknown_categories_ = task->GetCounter("unknown_categories");
    num_templates_ = task->GetCounter("wiki_templates");
    num_unknown_templates_ = task->GetCounter("unknown_templates");
    for (int i = 0; i < kNumAliasSources; ++i) {
      num_aliases_[i] =
          task->GetCounter(StrCat(kAliasSourceName[i], "_aliases"));
      num_discarded_aliases_[i] =
          task->GetCounter(StrCat(kAliasSourceName[i], "_discarded_aliases"));
    }

    // Load template repository configuration.
    Frame template_config(commons_, "/wp/templates/" + language_);
    if (template_config.valid()) {
      LOG(INFO) << "Loading template configuration";
      templates_.Init(this, template_config);
    }

    // Output aliases for all redirects.
    aliases_ = task->GetSink("aliases");
    for (Handle redirect : wikimap_.redirects()) {
      WikipediaMap::PageInfo page;
      wikimap_.GetRedirectInfo(redirect, &page);
      if (page.title.empty()) continue;
      if (page.qid.empty()) continue;
      if (page.type != WikipediaMap::ARTICLE) continue;

      string name;
      string disambiguation;
      Wiki::SplitTitle(page.title.str(), &name, &disambiguation);
      OutputAlias(page.qid, name, SRC_WIKIPEDIA_REDIRECT);
    }

    // Get output channel for parsed category documents.
    categories_ = task->GetSink("categories");
  }

  void Flush(task::Task *task) override {
    if (docnames_) {
      docnames_->Release();
      docnames_ = nullptr;
    }
  }

  void Process(Slice key, const Frame &frame) override {
    // Look up Wikidata page information in mapping.
    WikipediaMap::PageInfo page;
    if (!wikimap_.GetPageInfo(frame.Id(), &page) ||
        page.type == WikipediaMap::UNKNOWN) {
      VLOG(4) << "Unknown page: " << frame.Id();
      num_unknown_pages_->Increment();
      return;
    }

    // Convert article to document.
    ProcessArticle(frame, page.qid);
    Document document(frame, docnames_);

    // Handle the different page types.
    switch (page.type) {
      case WikipediaMap::ARTICLE:
        // Article: parse, extract aliases, output anchors as aliases for links.
        OutputTitleAlias(document);
        OutputAnchorAliases(document);
        OutputLinkAliases(document);
        Output(page.qid, frame);
        num_article_pages_->Increment();
        break;

      case WikipediaMap::CATEGORY:
        // Category: parse article, output anchors as aliases.
        OutputAnchorAliases(document);
        OutputLinkAliases(document);
        if (categories_ != nullptr) {
          categories_->Send(task::CreateMessage(page.qid, frame));
        }
        num_category_pages_->Increment();
        break;

      case WikipediaMap::DISAMBIGUATION:
        // Disambiguation: output aliases for all links.
        OutputDisambiguationAliases(document);
        num_disambiguation_pages_->Increment();
        break;

      case WikipediaMap::LIST:
        // Only keep anchor aliases from list pages.
        OutputAnchorAliases(document);
        OutputLinkAliases(document);
        num_list_pages_->Increment();
        break;

      default:
        VLOG(3) << "Unexpected page: " << page.type << " " << frame.Id();
        num_unexpected_pages_->Increment();
    }
  }

  void ProcessArticle(const Frame &page, Text qid) {
    // Parse Wikipedia article.
    string wikitext = page.GetString(n_page_text_);
    WikiParser parser(wikitext.c_str());
    parser.Parse();
    num_wiki_ast_nodes_->Increment(parser.nodes().size());

    // Extract annotations from article.
    WikiAnnotator annotator(page.store(), this);
    annotator.set_templates(&templates_);
    WikiExtractor extractor(parser);
    extractor.set_skip_tables(skip_tables_);
    extractor.Extract(&annotator);

    // Add basic document information.
    Builder article(page);
    article.AddLink(n_page_item_, qid);
    article.AddIsA(docnames_->n_document);
    string title = page.GetString(n_page_title_);
    article.Add(docnames_->n_url, Wiki::URL(language_, title));
    article.Add(docnames_->n_title, page.GetHandle(n_page_title_));
    const string &text =  annotator.text();
    article.Add(docnames_->n_text, text);
    article.Update();
    num_article_text_bytes_->Increment(text.size());

    // Tokenize article and add extracted annotations to document.
    Document document(page, docnames_);
    tokenizer_.Tokenize(&document);
    annotator.AddToDocument(&document);
    num_article_tokens_->Increment(document.num_tokens());
    document.Update();

    // Output aliases from extractor.
    for (const auto &alias : annotator.aliases()) {
      OutputAlias(qid, alias.name, alias.source);
    }
  }

  // Output alias for article title.
  void OutputTitleAlias(const Document &document) {
    string qid = document.top().GetFrame(n_page_item_).Id().str();
    string title = document.top().GetString(n_page_title_);
    string name;
    string disambiguation;
    Wiki::SplitTitle(title, &name, &disambiguation);
    OutputAlias(qid, name, SRC_WIKIPEDIA_TITLE);
  }

  // Output aliases for anchors.
  void OutputAnchorAliases(const Document &document) {
    for (int i = 0; i < document.num_spans(); ++i) {
      Span *span = document.span(i);
      string anchor = span->GetText();
      Text qid = span->Evoked().Id();
      OutputAlias(qid, anchor, SRC_WIKIPEDIA_ANCHOR);
    }
  }

  // Output aliases for links outside text.
  void OutputLinkAliases(const Document &document) {
    for (Handle h : document.themes()) {
      Frame theme(document.store(), h);
      if (theme.IsA(n_link_)) {
        Text qid = theme.GetFrame(Handle::is()).Id();
        Text anchor = theme.GetText(n_name_);
        OutputAlias(qid, anchor, SRC_WIKIPEDIA_LINK);
      }
    }
  }

  // Output aliases for disambiguation.
  void OutputDisambiguationAliases(const Document &document) {
    string title = document.title().str();
    if (title.empty()) return;
    string name;
    string disambiguation;
    Wiki::SplitTitle(title, &name, &disambiguation);
    for (int i = 0; i < document.num_spans(); ++i) {
      Span *span = document.span(i);
      string anchor = span->GetText();
      Text qid = span->Evoked().Id();
      OutputAlias(qid, name, SRC_WIKIPEDIA_DISAMBIGUATION);
    }
  }

  // Output alias.
  void OutputAlias(Text qid, Text name, AliasSource source) {
    if (!qid.empty() && !name.empty() && name.size() < 100) {
      string value = StrCat(source, ":", name);
      aliases_->Send(new task::Message(qid.slice(), Slice(value)));
      num_aliases_[source]->Increment();
    } else {
      num_discarded_aliases_[source]->Increment();
    }
  }

  // Link resolution interface for annotator.
  Text ResolveLink(Text link) override {
    num_links_->Increment();
    if (link.find('#') != -1) {
      num_fragment_links_->Increment();
      return Text();
    }
    Text qid = wikimap_.LookupLink(language_, link, WikipediaMap::ARTICLE);
    if (qid.empty()) num_dead_links_->Increment();
    return qid;
  }

  Text ResolveTemplate(Text link) override {
    WikipediaMap::PageInfo info;
    if (!wikimap_.GetPageInfo(language_, template_prefix_, link, &info)) {
      num_unknown_templates_->Increment();
      return Text();
    }
    if (info.type != WikipediaMap::TEMPLATE &&
        info.type != WikipediaMap::INFOBOX) {
      num_unknown_templates_->Increment();
      return Text();
    }
    num_templates_->Increment();
    return info.qid;
  }

  Text ResolveCategory(Text link) override {
    Text qid = wikimap_.LookupLink(language_, category_prefix_, link,
                                   WikipediaMap::CATEGORY);
    num_categories_->Increment();
    if (qid.empty()) num_unknown_categories_->Increment();
    return qid;
  }

 private:
  // Language.
  string language_;
  Handle lang_;
  string category_prefix_;
  string template_prefix_;

  // Mapping from Wikipedia ids to Wikidata ids.
  WikipediaMap wikimap_;

  // Plain text tokenizer.
  nlp::DocumentTokenizer tokenizer_;

  // Template macro repository.
  WikiTemplateRepository templates_;

  // Channel for aliases.
  task::Channel *aliases_ = nullptr;

  // Channel for categories.
  task::Channel *categories_ = nullptr;

  // Symbols.
  DocumentNames *docnames_ = nullptr;
  Name n_name_{names_, "name"};
  Name n_lang_category_{names_, "/lang/wikilang/wiki_category"};
  Name n_lang_template_{names_, "/lang/wikilang/wiki_template"};

  Name n_page_text_{names_, "/wp/page/text"};
  Name n_page_title_{names_, "/wp/page/title"};
  Name n_page_category_{names_, "/wp/page/category"};
  Name n_page_item_{names_, "/wp/page/item"};
  Name n_link_{names_, "/wp/link"};
  Name n_redirect_{names_, "/wp/redirect"};

  // Skip tables in Wikipedia documents.
  bool skip_tables_ = false;

  // Statistics.
  task::Counter *num_article_pages_;
  task::Counter *num_category_pages_;
  task::Counter *num_disambiguation_pages_;
  task::Counter *num_list_pages_;
  task::Counter *num_unknown_pages_;
  task::Counter *num_unexpected_pages_;
  task::Counter *num_wiki_ast_nodes_;
  task::Counter *num_article_text_bytes_;
  task::Counter *num_article_tokens_;
  task::Counter *num_links_;
  task::Counter *num_dead_links_;
  task::Counter *num_fragment_links_;
  task::Counter *num_categories_;
  task::Counter *num_unknown_categories_;
  task::Counter *num_templates_;
  task::Counter *num_unknown_templates_;

  task::Counter *num_aliases_[kNumAliasSources];
  task::Counter *num_discarded_aliases_[kNumAliasSources];
};

REGISTER_TASK_PROCESSOR("wikipedia-document-builder", WikipediaDocumentBuilder);

// Collect the aliases extracted from the Wikipedia document builder and build
// an alias profile for each entity.
class WikipediaAliasReducer : public task::Reducer {
 public:
  void Start(task::Task *task) override {
    Reducer::Start(task);
    string lang = task->Get("language", "en");
    language_ = commons_.Lookup(StrCat("/lang/" + lang));
    names_.Bind(&commons_);
    commons_.Freeze();
  }

  void Reduce(const task::ReduceInput &input) override {
    // Collect all the aliases for the item.
    Text qid = input.key();
    std::unordered_map<string, std::pair<int, int>> aliases;
    for (task::Message *message : input.messages()) {
      // Parse message value (<source>:<alias>).
      Text value = message->value();
      int colon = value.find(':');
      CHECK_NE(colon, -1);
      int source;
      CHECK(strtoint(value.substr(0, colon), &source));
      string name = value.substr(colon + 1).str();

      // Add alias to alias table.
      auto &item = aliases[name];
      item.first += 1;
      item.second |= 1 << source;
    }

    // Build alias profile.
    Store store(&commons_);
    Builder profile(&store);
    for (auto &a : aliases) {
      Builder alias(&store);
      alias.Add(n_name_, a.first);
      alias.Add(n_lang_, language_);
      alias.Add(n_sources_, a.second.second);
      alias.Add(n_count_, a.second.first);
      profile.Add(n_alias_, alias.Create());
    }
    Frame alias_profile  = profile.Create();

    // Output alias profile.
    Output(input.shard(), task::CreateMessage(qid, alias_profile));
  }

 private:
  // Commons store.
  Store commons_;

  // Symbols.
  Names names_;
  Name n_name_{names_, "name"};
  Name n_lang_{names_, "lang"};
  Name n_alias_{names_, "alias"};
  Name n_count_{names_, "count"};
  Name n_sources_{names_, "sources"};

  // Language.
  Handle language_;
};

REGISTER_TASK_PROCESSOR("wikipedia-alias-reducer", WikipediaAliasReducer);

// Extract categories from Wikipedia documents.
class CategoryItemExtractor : public task::FrameProcessor {
 public:
  void Process(Slice key, const Frame &frame) override {
    // Collect categories for page.
    Builder categories(frame.store());
    int num_categories = 0;
    for (const Slot &slot : frame) {
      if (slot.name == n_page_category_) {
        categories.Add(n_category_, slot.value);
        num_categories++;
      }
    }
    if (num_categories > 0) {
      Output(key, categories.Create());
    }
  }

 private:
  // Symbols.
  Name n_page_category_{names_, "/wp/page/category"};
  Name n_category_{names_, "/w/item/category"};
};

REGISTER_TASK_PROCESSOR("category-item-extractor", CategoryItemExtractor);

// Merge categories from different Wikipedias.
class CategoryItemMerger : public task::Reducer {
 public:
  void Start(task::Task *task) override {
    task::Reducer::Start(task);
    CHECK(names_.Bind(&commons_));
    commons_.Freeze();
  }

  void Reduce(const task::ReduceInput &input) override {
    // Merge all categories.
    Store store(&commons_);
    Builder categories(&store);
    HandleSet seen;
    for (task::Message *message : input.messages()) {
      for (const Slot &slot : DecodeMessage(&store, message)) {
        if (slot.name == n_category_ && seen.count(slot.value) == 0) {
          categories.Add(slot.name, slot.value);
          seen.insert(slot.value);
        }
      }
    }

    // Output merged categories for item.
    Frame merged = categories.Create();
    Output(input.shard(), task::CreateMessage(input.key(), merged));
  }

 private:
  // Commons store.
  Store commons_;

  // Symbols.
  Names names_;
  Name n_category_{names_, "/w/item/category"};
};

REGISTER_TASK_PROCESSOR("category-item-merger", CategoryItemMerger);

// Invert category links by taking inputs of items with categories and
// outputting pairs of (category, member).
class CategoryInverter : public task::FrameProcessor {
 public:
  void Process(Slice key, const Frame &frame) override {
    for (const Slot &slot : frame) {
      if (slot.name == n_item_category_) {
        Frame category(frame.store(), slot.value);
        output_->Send(new task::Message(category.Id().slice(), key));
      }
    }
  }

 private:
  // Symbols.
  Name n_item_category_{names_, "/w/item/category"};
};

REGISTER_TASK_PROCESSOR("category-inverter", CategoryInverter);

// Merge category members.
class CategoryMemberMerger : public task::Reducer {
 public:
  void Start(task::Task *task) override {
    task::Reducer::Start(task);
    task->Fetch("threshold", &threshold_);
    CHECK(names_.Bind(&commons_));
    commons_.Freeze();
  }

  void Reduce(const task::ReduceInput &input) override {
    // Merge all categories members.
    Store store(&commons_);
    Builder members(&store);
    int num_members = 0;
    for (task::Message *message : input.messages()) {
      members.AddLink(n_item_member_, message->value());
      num_members++;
    }

    // Check if category should be skipped.
    if (threshold_ > 0 && num_members > threshold_) {
      LOG(WARNING) << "Skipping category " << input.key()
                   << " with " << num_members << " members";
      return;
    }

    // Output members for category.
    Output(input.shard(), task::CreateMessage(input.key(), members.Create()));
  }

 private:
  // Commons store.
  Store commons_;

  // Threshold for skipping categories with many members.
  int threshold_ = 0;

  // Symbols.
  Names names_;
  Name n_item_member_{names_, "/w/item/member"};
};

REGISTER_TASK_PROCESSOR("category-member-merger", CategoryMemberMerger);

}  // namespace nlp
}  // namespace sling

