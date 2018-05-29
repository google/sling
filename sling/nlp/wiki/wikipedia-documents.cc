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
#include "sling/nlp/document/text-tokenizer.h"
#include "sling/nlp/document/tokens.h"
#include "sling/nlp/wiki/wikipedia-map.h"
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
class WikipediaDocumentBuilder : public task::FrameProcessor {
 public:
  // Language information for Wikipedia.
  struct LanguageInfo {
    Handle id;
    Text code;
    Text category_prefix;
    Text template_prefix;
  };

  void Startup(task::Task *task) override {
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

    // Initialize tokenizer.
    tokenizer_.InitLDC();

    // Get counters.
    num_article_pages_ = task->GetCounter("article_pages");
    num_disambiguation_pages_ = task->GetCounter("disambiguation_pages");
    num_list_pages_ = task->GetCounter("list_pages");
    num_unknown_pages_ = task->GetCounter("unknown_wikipedia_pages");
    num_unexpected_pages_ = task->GetCounter("unexpected_pages");
    num_wiki_ast_nodes_ = task->GetCounter("wiki_ast_nodes");
    num_article_text_bytes_ = task->GetCounter("article_text_bytes");
    num_article_tokens_ = task->GetCounter("article_tokens");
    num_images_ = task->GetCounter("wiki_images");
    num_links_ = task->GetCounter("wiki_links");
    num_dead_links_ = task->GetCounter("dead_links");
    num_fragment_links_ = task->GetCounter("fragment_links");
    num_anchors_ = task->GetCounter("wiki_anchors");
    num_templates_ = task->GetCounter("wiki_templates");
    num_special_templates_ = task->GetCounter("special_templates");
    num_infoboxes_ = task->GetCounter("wiki_infoboxes");
    num_unknown_templates_ = task->GetCounter("unknown_templates");
    num_categories_ = task->GetCounter("wiki_categories");
    num_unknown_categories_ = task->GetCounter("unknown_categories");
    num_empty_phrases_ = task->GetCounter("empty_phrases");
    for (int i = 0; i < kNumAliasSources; ++i) {
      num_aliases_[i] =
          task->GetCounter(StrCat(kAliasSourceName[i], "_aliases"));
      num_discarded_aliases_[i] =
          task->GetCounter(StrCat(kAliasSourceName[i], "_discarded_aliases"));
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
  }

  void Process(Slice key, const Frame &frame) override {
    // Look up Wikidata page information in mapping.
    WikipediaMap::PageInfo page;
    if (!wikimap_.GetPageInfo(frame.Id(), &page)) {
      VLOG(4) << "Unknown page: " << frame.Id();
      num_unknown_pages_->Increment();
      return;
    }

    // Handle the different page types.
    switch (page.type) {
      case WikipediaMap::ARTICLE:
        // Article: parse, extract aliases, output anchors as aliases for links.
        ProcessArticle(frame, page.qid);
        OutputTitleAlias(frame);
        OutputAnchorAliases(frame);
        Output(page.qid, frame);
        num_article_pages_->Increment();
        break;

      case WikipediaMap::DISAMBIGUATION:
        // Disambiguation: output aliases for all links.
        ProcessArticle(frame, page.qid);
        OutputDisambiguationAliases(frame);
        num_disambiguation_pages_->Increment();
        break;

      case WikipediaMap::LIST:
        // Only keep anchor aliases from list pages.
        ProcessArticle(frame, page.qid);
        OutputAnchorAliases(frame);
        num_list_pages_->Increment();
        break;

      default:
        VLOG(3) << "Unexpected page: " << page.type << " " << frame.Id();
        num_unexpected_pages_->Increment();
    }
  }

  void ProcessArticle(const Frame &page, Text qid) {
    // Get language for article.
    LanguageInfo lang;
    GetLanguageInfo(page.GetHandle(n_lang_), &lang);

    // Parse Wikipedia article.
    Store *store = page.store();
    string wikitext = page.GetString(n_page_text_);
    WikiParser parser(wikitext.c_str());
    parser.Parse();
    parser.Extract();

    // Add basic document information.
    Builder article(page);
    article.AddLink(n_page_item_, qid);
    article.AddIsA(n_document_);
    string title = page.GetString(n_page_title_);
    article.Add(n_document_url_, Wiki::URL(lang.code.str(), title));
    article.Add(n_document_title_, page.GetHandle(n_page_title_));

    // Add document text and tokens.
    const string &text =  parser.text();
    article.Add(n_document_text_, text);
    Handles tokens(store);
    tokenizer_.Tokenize(text,
      [this, store, &tokens](const nlp::Tokenizer::Token &t) {
        int index = tokens.size();
        Builder token(store);
        token.AddIsA(n_token_);
        token.Add(n_token_index_, index);
        token.Add(n_token_text_, t.text);
        token.Add(n_token_start_, t.begin);
        token.Add(n_token_length_, t.end - t.begin);
        if (t.brk != nlp::SPACE_BREAK) {
          token.Add(n_token_break_, t.brk);
        }
        tokens.push_back(token.Create().handle());
      }
    );
    Array token_array(store, tokens);
    article.Add(n_document_tokens_, token_array);

    // Add links as mentions.
    nlp::Tokens document_tokens(token_array);
    num_wiki_ast_nodes_->Increment(parser.num_ast_nodes());
    num_article_text_bytes_->Increment(parser.text().size());
    num_article_tokens_->Increment(tokens.size());
    for (const auto &node : parser.nodes()) {
      switch (node.type) {
        case WikiParser::IMAGE:
          num_images_->Increment();
          break;

        case WikiParser::TEMPLATE: {
          num_templates_->Increment();
          if (node.param != 0) {
            num_special_templates_->Increment();
          } else {
            WikipediaMap::PageInfo tmpl;
            if (GetTemplateInfo(lang, node.name(), &tmpl)) {
              if (tmpl.type == WikipediaMap::INFOBOX) {
                num_infoboxes_->Increment();
              }
            } else {
              VLOG(8) << "Unknown template: " << node.name();
              num_unknown_templates_->Increment();
            }
          }
          break;
        }

        case WikiParser::CATEGORY: {
          num_categories_->Increment();
          Text category = LookupCategory(lang, node.name());
          if (category.empty()) {
            VLOG(7) << "Unknown category: " << node.name();
            num_unknown_categories_->Increment();
          } else {
            article.AddLink(n_page_category_, category);
          }
          break;
        }

        case WikiParser::LINK: {
          num_links_->Increment();
          Text linkname = node.name();
          if (linkname.find('#') != -1) {
            num_fragment_links_->Increment();
          } else {
            Text link = LookupLink(lang, linkname);
            if (link.empty()) {
              VLOG(9) << "Dead link: " << node.name();
              num_dead_links_->Increment();
            } else if (node.anchored()) {
              // Get tokens span.
              int begin = document_tokens.Locate(node.text_begin);
              int end = document_tokens.Locate(node.text_end);
              int length = end - begin;

              if (begin == -1 || begin == end) {
                num_empty_phrases_->Increment();
              } else {
                // Add mention with link.
                Builder mention(store);
                mention.AddIsA(n_link_);
                mention.Add(n_phrase_begin_, begin);
                if (length != 1) mention.Add(n_phrase_length_, length);
                mention.Add(n_name_, document_tokens.Phrase(begin, end));
                mention.AddLink(n_phrase_evokes_, link);
                article.Add(n_document_mention_, mention.Create());
              }
              num_anchors_->Increment();
            }
          }
          break;
        }

        default: break;
      }
    }

    // Update article document.
    article.Update();
  }

  // Output alias for article title.
  void OutputTitleAlias(const Frame &document) {
    string qid = document.GetFrame(n_page_item_).Id().str();
    string title = document.GetString(n_page_title_);
    string name;
    string disambiguation;
    Wiki::SplitTitle(title, &name, &disambiguation);
    OutputAlias(qid, name, SRC_WIKIPEDIA_TITLE);
  }

  // Output aliases for anchors.
  void OutputAnchorAliases(const Frame &document) {
    nlp::Tokens tokens(document);
    for (const Slot &s : document) {
      if (s.name != n_document_mention_) continue;
      Frame mention(document.store(), s.value);
      int begin = mention.GetInt(n_phrase_begin_, 0);
      int length = mention.GetInt(n_phrase_length_, 1);
      string anchor = tokens.Phrase(begin, begin + length);
      for (const Slot &e : mention) {
        if (e.name != n_phrase_evokes_) continue;
        Text qid = Frame(document.store(), e.value).Id();
        OutputAlias(qid, anchor, SRC_WIKIPEDIA_ANCHOR);
      }
    }
  }

  // Output aliases for disambiguation.
  void OutputDisambiguationAliases(const Frame &document) {
    string title = document.GetString(n_document_title_);
    if (title.empty()) return;
    string name;
    string disambiguation;
    Wiki::SplitTitle(title, &name, &disambiguation);
    for (const Slot &s : document) {
      if (s.name != n_document_mention_) continue;
      Frame mention(document.store(), s.value);
      for (const Slot &e : mention) {
        if (e.name != n_phrase_evokes_) continue;
        Text qid = Frame(document.store(), e.value).Id();
        OutputAlias(qid, name, SRC_WIKIPEDIA_DISAMBIGUATION);
      }
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

  // Get language-specific information for Wikipedia.
  void GetLanguageInfo(Handle lang, LanguageInfo *info) {
    Frame langinfo(commons_, lang);
    CHECK(langinfo.valid());
    info->id = lang;
    info->code = langinfo.GetText(n_lang_code_);
    info->category_prefix = langinfo.GetText(n_lang_category_);
    info->template_prefix = langinfo.GetText(n_lang_template_);
  }

  // Lookup Wikidata id for link.
  Text LookupLink(const LanguageInfo &lang, Text name) {
    return wikimap_.LookupLink(lang.code, name);
  }

  // Get page info for template.
  bool GetTemplateInfo(const LanguageInfo &lang, Text name,
                       WikipediaMap::PageInfo *info) {
    return wikimap_.GetPageInfo(lang.code, lang.template_prefix, name, info);
  }

  // Lookup Wikidata id for category.
  Text LookupCategory(const LanguageInfo &lang, Text name) {
    return wikimap_.LookupLink(lang.code, lang.category_prefix, name);
  }

 private:
  // Mapping from Wikipedia ids to Wikidata ids.
  WikipediaMap wikimap_;

  // Plain text tokenizer.
  nlp::Tokenizer tokenizer_;

  // Channel for aliases.
  task::Channel *aliases_ = nullptr;

  // Symbols.
  Name n_name_{names_, "name"};
  Name n_lang_{names_, "lang"};
  Name n_lang_code_{names_, "code"};
  Name n_lang_category_{names_, "/lang/wikilang/wiki_category"};
  Name n_lang_template_{names_, "/lang/wikilang//wiki_template"};

  Name n_page_text_{names_, "/wp/page/text"};
  Name n_page_title_{names_, "/wp/page/title"};
  Name n_page_category_{names_, "/wp/page/category"};
  Name n_page_item_{names_, "/wp/page/item"};
  Name n_link_{names_, "/wp/link"};
  Name n_redirect_{names_, "/wp/redirect"};

  Name n_document_{names_, "/s/document"};
  Name n_document_title_{names_, "/s/document/title"};
  Name n_document_url_{names_, "/s/document/url"};
  Name n_document_text_{names_, "/s/document/text"};
  Name n_document_tokens_{names_, "/s/document/tokens"};
  Name n_document_mention_{names_, "/s/document/mention"};

  Name n_token_{names_, "/s/token"};
  Name n_token_index_{names_, "/s/token/index"};
  Name n_token_text_{names_, "/s/token/text"};
  Name n_token_start_{names_, "/s/token/start"};
  Name n_token_length_{names_, "/s/token/length"};
  Name n_token_break_{names_, "/s/token/break"};

  Name n_phrase_begin_{names_, "/s/phrase/begin"};
  Name n_phrase_length_{names_, "/s/phrase/length"};
  Name n_phrase_evokes_{names_, "/s/phrase/evokes"};

  // Statistics.
  task::Counter *num_article_pages_;
  task::Counter *num_disambiguation_pages_;
  task::Counter *num_list_pages_;
  task::Counter *num_unknown_pages_;
  task::Counter *num_unexpected_pages_;
  task::Counter *num_wiki_ast_nodes_;
  task::Counter *num_article_text_bytes_;
  task::Counter *num_article_tokens_;
  task::Counter *num_images_;
  task::Counter *num_links_;
  task::Counter *num_dead_links_;
  task::Counter *num_fragment_links_;
  task::Counter *num_anchors_;
  task::Counter *num_templates_;
  task::Counter *num_special_templates_;
  task::Counter *num_infoboxes_;
  task::Counter *num_unknown_templates_;
  task::Counter *num_categories_;
  task::Counter *num_unknown_categories_;
  task::Counter *num_empty_phrases_;

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
      alias.Add(n_alias_sources_, a.second.second);
      alias.Add(n_alias_count_, a.second.first);
      profile.Add(n_profile_alias_, alias.Create());
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
  Name n_profile_alias_{names_, "/s/profile/alias"};
  Name n_alias_count_{names_, "/s/alias/count"};
  Name n_alias_sources_{names_, "/s/alias/sources"};

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

}  // namespace nlp
}  // namespace sling

