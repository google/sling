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

#include <iostream>
#include <string>

#include "sling/base/flags.h"
#include "sling/base/init.h"
#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/file/file.h"
#include "sling/frame/serialization.h"
#include "sling/frame/store.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/document/document-tokenizer.h"
#include "sling/nlp/wiki/wiki.h"
#include "sling/nlp/wiki/wiki-annotator.h"
#include "sling/nlp/wiki/wiki-extractor.h"
#include "sling/nlp/wiki/wiki-parser.h"
#include "sling/nlp/wiki/wikipedia-map.h"

DEFINE_string(input, "test.txt", "input file with wiki text");
DEFINE_string(lang, "", "language for wiki text");
DEFINE_bool(ann, false, "output document annotations");

using namespace sling;
using namespace sling::nlp;

class Resolver : public WikiLinkResolver {
 public:
  void Init() {
    string dir = "local/data/e/wiki/" + FLAGS_lang;
    wikimap_.LoadRedirects(dir + "/redirects.sling");
    wikimap_.LoadMapping(dir + "/mapping.sling");
  }

  Text ResolveLink(Text link) override {
    if (link.find('#') != -1) return Text();
    return wikimap_.LookupLink(FLAGS_lang, link, WikipediaMap::ARTICLE);
  }

  Text ResolveTemplate(Text link) override {
    WikipediaMap::PageInfo info;
    if (!wikimap_.GetPageInfo(FLAGS_lang, "Template", link, &info)) {
      return Text();
    }
    if (info.type != WikipediaMap::TEMPLATE &&
        info.type != WikipediaMap::INFOBOX) {
      return Text();
    }
    return info.qid;
  }

  Text ResolveCategory(Text link) override {
    return wikimap_.LookupLink(FLAGS_lang, "Category", link,
                               WikipediaMap::CATEGORY);
  }

  Store *store() { return wikimap_.store(); }

 private:
  WikipediaMap wikimap_;
};

int main(int argc, char *argv[]) {
  InitProgram(&argc, &argv);

  Resolver resolver;
  Store *store = resolver.store();
  if (!FLAGS_lang.empty()) {
    resolver.Init();
    LoadStore("data/wiki/calendar.sling", store);
    LoadStore("data/wiki/countries.sling", store);
    LoadStore("data/wiki/templates-" + FLAGS_lang + ".sling", store);
    LoadStore("data/wiki/units.sling", store);
  }

  string wikitext;
  CHECK(File::ReadContents(FLAGS_input, &wikitext));

  WikiParser parser(wikitext.c_str());
  parser.Parse();

  WikiExtractor extractor(parser);
  WikiAnnotator annotator(store, &resolver);
  Frame template_config(store, "/wp/templates/" + FLAGS_lang);
  WikiTemplateRepository templates;
  if (template_config.valid()) {
    templates.Init(&resolver, template_config);
    annotator.set_templates(&templates);
  }

  extractor.Extract(&annotator);

  Document document(store);
  if (FLAGS_ann) {
    DocumentTokenizer tokenizer;
    document.SetText(annotator.text());
    tokenizer.Tokenize(&document);
    annotator.AddToDocument(&document);
    document.Update();
  }

  std::cout << "<html>\n";
  std::cout << "<head>\n";
  std::cout << "<meta charset='utf-8'/>\n";
  std::cout << "</head>\n";
  std::cout << "<body>\n";
  std::cout <<  annotator.text() << "\n";
  if (FLAGS_ann) {
    std::cout << "<h1>Mentions</h1>\n";
    std::cout << "<table border=1>\n";
    std::cout << "<tr><th>Phrase</th><th>Annotations</th></tr>\n";
    for (int i = 0; i < document.num_spans(); ++i) {
      Span *span = document.span(i);
      std::cout << "<tr><td>" << span->GetText() << "</td><td><pre>\n";
      Handles evoked(store);
      span->AllEvoked(&evoked);
      for (Handle h : evoked) {
        std::cout << ToText(store, h) << "\n";
      }
      std::cout << "</pre></td></tr>\n";
    }
    std::cout << "</table>\n";
    if (!document.themes().empty()) {
      std::cout << "<h1>Themes</h1>\n";
      for (Handle theme : document.themes()) {
        std::cout << "<pre>\n";
        std::cout << ToText(document.store(), theme, 2);
        std::cout << "</pre>\n";
      }
    }
  }
  std::cout << "<h1>AST</h1>\n<pre>\n";
  parser.PrintAST(0, 0);
  std::cout << "</pre>\n";
  std::cout << "</body></html>\n";

  return 0;
}

