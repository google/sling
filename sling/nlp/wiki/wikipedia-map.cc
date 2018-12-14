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

#include "sling/nlp/wiki/wikipedia-map.h"

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/frame/object.h"
#include "sling/frame/serialization.h"
#include "sling/frame/store.h"
#include "sling/string/text.h"

namespace sling {
namespace nlp {

WikipediaMap::WikipediaMap() {
  // Look up symbols.
  n_qid_ = store_.Lookup("/w/item/qid");
  n_kind_ = store_.Lookup("/w/item/kind");
  n_redirect_ = store_.Lookup("/wp/redirect");
  n_redirect_title_ = store_.Lookup("/wp/redirect/title");
  n_redirect_link_ = store_.Lookup("/wp/redirect/link");

  // Initialize page type mapping.
  typemap_[store_.Lookup("/w/item/kind/article")] = ARTICLE;
  typemap_[store_.Lookup("/w/item/kind/disambiguation")] = DISAMBIGUATION;
  typemap_[store_.Lookup("/w/item/kind/category")] = CATEGORY;
  typemap_[store_.Lookup("/w/item/kind/list")] = LIST;
  typemap_[store_.Lookup("/w/item/kind/template")] = TEMPLATE;
  typemap_[store_.Lookup("/w/item/kind/infobox")] = INFOBOX;

  // Allow duplicate symbols.
  // TODO(ringgaard): Investigate if this is really needed, and if so, document
  // why it is needed.
  options_.symbol_rebinding = true;
}

void WikipediaMap::LoadMapping(Text filename) {
  // Load the whole mapping into the mapping store.
  store_.LockGC();
  FileDecoder decoder(&store_, filename.str());
  decoder.DecodeAll();
  store_.UnlockGC();
}

void WikipediaMap::LoadRedirects(Text filename) {
  // Load the redirects into the mapping store and make list of redirects.
  store_.LockGC();
  FileDecoder decoder(&store_, filename.str());
  while (!decoder.done()) {
    redirects_.push_back(decoder.DecodeObject());
  }
  store_.UnlockGC();
}

Handle WikipediaMap::Resolve(Handle h) {
  // Recursively resolve redirects.
  static const int MAX_REDIRECTS = 32;
  const FrameDatum *frame = store_.GetFrame(h);
  int num_redirects = 0;
  while (true) {
    Handle redir = frame->get(n_redirect_link_);
    if (redir.IsNil()) break;
    if (redir == frame->self) break;
    frame = store_.GetFrame(redir);
    if (++num_redirects > MAX_REDIRECTS) {
      LOG(WARNING) << "Unresolved redirect from " << store_.DebugString(h)
                   << " ending at " << store_.DebugString(redir);
      return h;
    }
  }
  return frame->self;
}

Text WikipediaMap::Lookup(Text id, PageType type) {
  // Look up item in mapping.
  Handle h = store_.LookupExisting(id);
  if (h.IsNil()) return Text();

  // Resolve redirects.
  h = Resolve(h);

  // Get Wikidata id for item.
  Frame frame(&store_, h);

  // Check target type.
  if (type != UNKNOWN) {
    auto f = typemap_.find(frame.GetHandle(n_kind_));
    if (f == typemap_.end() || f->second != type) return Text();
  }

  // Return qid.
  Frame qid = frame.GetFrame(n_qid_);
  if (qid.invalid()) return Text();
  return qid.Id();
}

bool WikipediaMap::GetPageInfo(Text id, PageInfo *info) {
  // Lookup id in mapping.
  Frame item(&store_, id);
  if (item.invalid() || item.IsProxy()) return false;

  // Resolve redirects.
  info->source_id = item.Id();
  if (item.IsA(n_redirect_)) {
   info->title = item.GetText(n_redirect_title_);
   item = Frame(&store_, Resolve(item.handle()));
  }

  // Return page information from target item.
  info->target_id = item.Id();
  auto f = typemap_.find(item.GetHandle(n_kind_));
  info->type = f == typemap_.end() ? UNKNOWN : f->second;
  info->qid = item.Get(n_qid_).AsFrame().Id();

  return true;
}

bool WikipediaMap::GetPageInfo(Text lang, Text link, PageInfo *info) {
  string id = Wiki::Id(lang.str(), link.str());
  return GetPageInfo(id, info);
}

bool WikipediaMap::GetPageInfo(Text lang,
                               Text prefix,
                               Text link,
                               PageInfo *info) {
  string id = Wiki::Id(lang.str(), prefix.str(), link.str());
  return GetPageInfo(id, info);
}

void WikipediaMap::GetRedirectInfo(Handle redirect, PageInfo *info) {
  // Get redirect from store.
  Frame item(&store_, redirect);

  // Resolve redirects.
  info->source_id = item.Id();
  if (item.IsA(n_redirect_)) {
   info->title = item.GetText(n_redirect_title_);
   item = Frame(&store_, Resolve(item.handle()));
  }

  // Return page information from target item.
  info->target_id = item.Id();
  if (item.IsA(n_redirect_)) {
    info->type = REDIRECT;
  } else {
    auto f = typemap_.find(item.GetHandle(n_kind_));
    info->type = f == typemap_.end() ? UNKNOWN : f->second;
  }
  info->qid = item.Get(n_qid_).AsFrame().Id();
}

Text WikipediaMap::LookupLink(Text lang, Text link, PageType type) {
  string id = Wiki::Id(lang.str(), link.str());
  return Lookup(id, type);
}

Text WikipediaMap::LookupLink(Text lang, Text prefix, Text link,
                              PageType type) {
  string id = Wiki::Id(lang.str(), prefix.str(), link.str());
  return Lookup(id, type);
}

}  // namespace nlp
}  // namespace sling

