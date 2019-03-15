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

#include <string>
#include <vector>

#include "sling/base/registry.h"
#include "sling/base/types.h"
#include "sling/frame/object.h"
#include "sling/nlp/document/document.h"
#include "sling/nlp/wiki/wiki.h"
#include "sling/nlp/wiki/wiki-extractor.h"
#include "sling/string/text.h"

namespace sling {
namespace nlp {

class WikiAnnotator;

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

// Wrapper around wiki template node.
class WikiTemplate {
 public:
  typedef WikiParser::Node Node;

  WikiTemplate(const Node &node, WikiExtractor *extractor)
      : node_(node), extractor_(extractor) {}

  // Return template name.
  Text name() const { return node_.name(); }

  // Return the number of positional (i.e. unnamed) arguments.
  int NumArgs() const;

  // Return node for named template argument, or null if it is not found.
  const Node *GetArgument(Text name) const;

  // Return node for positional template argument. First argument is 1.
  const Node *GetArgument(int index) const;

  // Return node for named or positional template argument.
  const Node *GetArgument(Text name, int index) const;

  // Get all template arguments.
  void GetArguments(std::vector<const Node *> *args) const;

  // Return plain text value for named or positional template argument.
  string GetValue(const Node *node) const;
  string GetValue(Text name) const { return GetValue(GetArgument(name)); }
  string GetValue(int index) const { return GetValue(GetArgument(index)); }

  // Return numeric value for named or positional template argument. Return
  // -1 if the argument does not exist or is not a number and return zero if
  // the argument is empty.
  int GetNumber(const Node *node) const;
  int GetNumber(Text name) const { return GetNumber(GetArgument(name)); }
  int GetNumber(int index) const { return GetNumber(GetArgument(index)); }

  // Return floating point value for named or positional template argument.
  float GetFloat(const Node *node) const;
  float GetFloat(Text name) const { return GetFloat(GetArgument(name)); }
  float GetFloat(int index) const { return GetFloat(GetArgument(index)); }

  // Extract text for template argument
  void Extract(const Node *node) const;
  void Extract(Text name) const { Extract(GetArgument(name)); }
  void Extract(int index) const { Extract(GetArgument(index)); }

  // Skip extraction for template argument
  void ExtractSkip(const Node *node) const;
  void ExtractSkip(Text name) const { ExtractSkip(GetArgument(name)); }
  void ExtractSkip(int index) const { ExtractSkip(GetArgument(index)); }

  // Check if a node is empty, i.e. only whitespace and comments.
  bool IsEmpty(const Node *node) const;

  // Return template extractor.
  WikiExtractor *extractor() const { return extractor_; }

 private:
  // Template node.
  const Node &node_;

  // Extractor for extracting template argument values.
  WikiExtractor *extractor_;
};

// A wiki macro processor is used for expanding wiki templates into text and
// annotations.
class WikiMacro : public Component<WikiMacro> {
 public:
  typedef WikiParser::Node Node;

  virtual ~WikiMacro() = default;

  // Initialize wiki macro processor from configuration.
  virtual void Init(const Frame &config) {}

  // Expand template by adding content and annotations to annotator.
  virtual void Generate(const WikiTemplate &templ, WikiAnnotator *annotator) {}

  // Extract annotations from unanchored template.
  virtual void Extract(const WikiTemplate &templ, WikiAnnotator *annotator) {}
};

#define REGISTER_WIKI_MACRO(type, component) \
    REGISTER_COMPONENT_TYPE(sling::nlp::WikiMacro, type, component)

// Repository of wiki macro configurations for a language for expanding wiki
// templates when processing a Wikipedia page.
class WikiTemplateRepository {
 public:
  ~WikiTemplateRepository();

  // Intialize repository from configuration.
  void Init(WikiLinkResolver *resolver, const Frame &frame);

  // Look up macro processor for temaplate name .
  WikiMacro *Lookup(Text name);

 private:
  // Store for templates.
  Store *store_ = nullptr;

  // Link resolver for looking up templates.
  WikiLinkResolver *resolver_ = nullptr;

  // Mapping from template frame to wiki macro procesor.
  HandleMap<WikiMacro *> repository_;
};

// Wiki extractor sink for collecting text and annotations for Wikipedia page.
// It collects text span information about evoked frames than can later be added
// to a SLING document when the text has been tokenized. It also collects
// thematic frames for unanchored annotations.
class WikiAnnotator : public WikiTextSink {
 public:
  // Alias for document topic.
  struct Alias {
    Alias(const string &name, AliasSource source)
        : name(name), source(source) {}

    string name;         // alias name
    AliasSource source;  // alias source
  };

  // Initialize document annotator. The frame annotations will be created in
  // the store and links will be resolved using the resolver.
  WikiAnnotator(Store *store, WikiLinkResolver *resolver);

  // Initialize sub-annotator based on another annotator. Please notice that
  // this is not a copy constructor.
  explicit WikiAnnotator(WikiAnnotator *other);

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

  // Add alias.
  void AddAlias(const string &name, AliasSource source);

  // Get list of aliases.
  const std::vector<Alias> &aliases() const { return aliases_; }

  // Return store for annotations.
  Store *store() { return store_; }

  // Return link resolver.
  WikiLinkResolver *resolver() { return resolver_; }

  // Get/set template repository.
  WikiTemplateRepository *templates() const { return templates_; }
  void set_templates(WikiTemplateRepository *templates) {
    templates_ = templates;
  }

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

  // Template generator.
  WikiTemplateRepository *templates_ = nullptr;

  // Annotated spans.
  Annotations annotations_;

  // Thematic frames.
  Handles themes_;

  // Categories.
  Handles categories_;

  // Aliases.
  std::vector<Alias> aliases_;

  // Symbols.
  Names names_;
  Name n_name_{names_, "name"};
  Name n_link_{names_, "/wp/link"};
  Name n_page_category_{names_, "/wp/page/category"};
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_WIKI_WIKI_ANNOTATOR_H_

