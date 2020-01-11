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

#include "sling/nlp/parser/frame-evaluation.h"

#include <algorithm>

#include "sling/base/logging.h"
#include "sling/frame/serialization.h"
#include "sling/nlp/document/document-corpus.h"
#include "sling/string/strcat.h"
#include "sling/string/printf.h"

namespace sling {
namespace nlp {

// Parallel corpus for file-based document source.
class FileParallelCorpus : public ParallelCorpus {
 public:
  // Open corpora.
  FileParallelCorpus(Store *commons,
                     const string &gold_file_pattern,
                     const string &test_file_pattern)
      :  commons_(commons),
         gold_corpus_(commons, gold_file_pattern),
         test_corpus_(commons, test_file_pattern) {}

  // Read next document pair from corpora.
  bool Next(Store **store, Document **golden, Document **predicted) override {
    *store = new Store(commons_);
    *golden = gold_corpus_.Next(*store);
    *predicted = test_corpus_.Next(*store);
    if (*golden == nullptr) {
      CHECK(*predicted == nullptr);
      delete *store;
      return false;
    } else {
      CHECK(*predicted != nullptr);
      return true;
    }
  }

 private:
  Store *commons_;               // commons store for documents
  DocumentCorpus gold_corpus_;   // corpus with gold annotations
  DocumentCorpus test_corpus_;   // corpus with predicted annotations
};

bool FrameEvaluation::Alignment::Map(Handle source, Handle target) {
  // Do not allow any previous mapping to be overwritten.
  if (!Lookup(source).IsNil()) return false;

  // Only allow target to be used once.
  if (!target.IsNil() && targets_.count(target) > 0) return false;

  // Add mapping to alignment.
  (*this)[source] = target;
  if (!target.IsNil()) targets_.insert(target);
  return true;
}

// Returns the target that frame is mapped to or nil.
Handle FrameEvaluation::Alignment::Lookup(Handle handle) const {
  auto f = find(handle);
  return f == end() ? Handle::nil() : f->second;
}

void FrameEvaluation::Evaluate(ParallelCorpus *corpus, Output *output) {
  // Benchmarks.
  auto &mention = output->mention;
  auto &frame = output->frame;
  auto &pair = output->pair;
  auto &edge = output->edge;
  auto &role = output->role;
  auto &type = output->type;
  auto &label = output->label;

  // Statistics counters.
  output->num_golden_spans = 0;
  output->num_predicted_spans = 0;
  output->num_golden_frames = 0;
  output->num_predicted_frames = 0;

  Store *store;
  Document *golden;
  Document *predicted;
  while (corpus->Next(&store, &golden, &predicted)) {
    CHECK_EQ(golden->num_tokens(), predicted->num_tokens());
    Frame golden_top = golden->top();
    Frame predicted_top = predicted->top();

    // Get mention maps.
    MentionMap golden_mentions;
    MentionMap predicted_mentions;
    GetMentionMap(golden_top, &golden_mentions);
    GetMentionMap(predicted_top, &predicted_mentions);

    // Compute mention span alignments.
    Alignment g2p_mention_alignment;
    Alignment p2g_mention_alignment;
    AlignMentions(golden_mentions,
                  predicted_mentions,
                  &g2p_mention_alignment);
    AlignMentions(predicted_mentions,
                  golden_mentions,
                  &p2g_mention_alignment);

    // Compute evoked frame alignment.
    Alignment g2p_frame_alignment;
    Alignment p2g_frame_alignment;
    AlignEvokes(store, g2p_mention_alignment, &g2p_frame_alignment);
    AlignEvokes(store, p2g_mention_alignment, &p2g_frame_alignment);

    // Align frames that are not directly evoked from a span.
    AlignFrames(store, &g2p_frame_alignment);
    AlignFrames(store, &p2g_frame_alignment);

    // Compute mention precision and recall.
    AlignmentAccuracy(g2p_mention_alignment, &mention.recall);
    AlignmentAccuracy(p2g_mention_alignment, &mention.precision);

    // Compute frame precision and recall.
    AlignmentAccuracy(g2p_frame_alignment, &frame.recall);
    AlignmentAccuracy(p2g_frame_alignment, &frame.precision);

    // Compute role precision and recall.
    RoleAccuracy(store, g2p_frame_alignment,
                 &pair.recall, &edge.recall, &role.recall,
                 &type.recall, &label.recall);
    RoleAccuracy(store, p2g_frame_alignment,
                 &pair.precision, &edge.precision, &role.precision,
                 &type.precision, &label.precision);

    // Update statistics.
    output->num_golden_spans += golden_mentions.size();
    output->num_predicted_spans += predicted_mentions.size();
    output->num_golden_frames += g2p_frame_alignment.size();
    output->num_predicted_frames += p2g_frame_alignment.size();

    delete golden;
    delete predicted;
    delete store;
  }

  // Compute the slot score as the sum of the type, role, and label scores.
  auto &slot = output->slot;
  slot.add(type);
  slot.add(role);
  slot.add(label);

  // Compute the combined score as the sum of the other scores.
  auto &combined = output->combined;
  combined.add(mention);
  combined.add(frame);
  combined.add(type);
  combined.add(role);
  combined.add(label);
}

void FrameEvaluation::Evaluate(Store *commons,
                               const string &gold_file_pattern,
                               const string &test_file_pattern,
                               FrameEvaluation::Output *output) {
  FileParallelCorpus corpus(commons, gold_file_pattern, test_file_pattern);
  Evaluate(&corpus, output);
}

std::vector<string> FrameEvaluation::EvaluateAndSummarize(
    const string &commons_file,
    const string &gold_file_pattern,
    const string &test_file_pattern) {
  Store commons;
  FileDecoder decoder(&commons, commons_file);
  decoder.DecodeAll();
  commons.Freeze();

  Output eval;
  Evaluate(&commons, gold_file_pattern, test_file_pattern, &eval);

  // Write output to output_file.
  Scores scores;
  eval.GetScores(&scores);

  std::vector<string> lines;
  for (auto &score : scores) {
    lines.emplace_back(StrCat(score.first, "\t", score.second));
  }
  return lines;
}

void FrameEvaluation::EvaluateAndWrite(const string &commons_file,
                                       const string &gold_file_pattern,
                                       const string &test_file_pattern,
                                       const string &output_file) {
  std::vector<string> summary =
      EvaluateAndSummarize(commons_file, gold_file_pattern, test_file_pattern);
  File *f = File::Open(output_file, "w");
  for (const string &line : summary) {
    f->WriteLine(line);
  }
  f->Close();
}

void FrameEvaluation::Benchmark::GetScores(const string &name,
                                           Scores *scores) const {
  double p = precision.accuracy();
  double r = recall.accuracy();
  double f1 = fscore();
  scores->emplace_back(StrCat(name, "_P+"), precision.correct);
  scores->emplace_back(StrCat(name, "_P-"), precision.wrong);
  scores->emplace_back(StrCat(name, "_R+"), recall.correct);
  scores->emplace_back(StrCat(name, "_R-"), recall.wrong);
  scores->emplace_back(StrCat(name, "_Precision"), p * 100.0);
  scores->emplace_back(StrCat(name, "_Recall"), r * 100.0);
  scores->emplace_back(StrCat(name, "_F1"), f1 * 100.0);
}

string FrameEvaluation::Benchmark::Summary() const {
  double p = precision.accuracy() * 100.0;
  double r = recall.accuracy() * 100.0;
  double f1 = fscore() * 100.0;
  return StringPrintf("P=%5.2f, R=%5.2f, F1=%5.2f", p, r, f1);
}

void FrameEvaluation::Output::GetScores(Scores *scores) const {
  mention.GetScores("SPAN", scores);
  frame.GetScores("FRAME", scores);
  pair.GetScores("PAIR", scores);
  edge.GetScores("EDGE", scores);
  role.GetScores("ROLE", scores);
  type.GetScores("TYPE", scores);
  label.GetScores("LABEL", scores);
  slot.GetScores("SLOT", scores);
  combined.GetScores("COMBINED", scores);
  scores->emplace_back("#GOLDEN_SPANS", num_golden_spans);
  scores->emplace_back("#PREDICTED_SPANS", num_predicted_spans);
  scores->emplace_back("#GOLDEN_FRAMES", num_golden_frames);
  scores->emplace_back("#PREDICTED_FRAMES", num_predicted_frames);
}

void FrameEvaluation::GetMentionMap(
    const Frame &frame, MentionMap *mentions) {
  Store *store = frame.store();
  Handle n_mention = store->Lookup("mention");
  Handle n_begin = store->Lookup("begin");
  Handle n_length = store->Lookup("length");

  for (const Slot &slot : frame) {
    if (slot.name ==  n_mention) {
      Frame mention(store, slot.value);
      int begin = mention.GetInt(n_begin);
      int length = mention.GetInt(n_length);
      int end = length == 0 ? begin + 1 : begin + length;
      FrameEvaluation::Span span(begin, end);
      (*mentions)[span] = mention.handle();
    }
  }
}

void FrameEvaluation::AlignMentions(const MentionMap &source,
                                    const MentionMap &target,
                                    Alignment *alignment) {
  // Iterate over all spans in source.
  for (const auto &s : source) {
    // Try to match matching span in target.
    auto f = target.find(s.first);
    if (f == target.end()) {
      // No matching span in b. Insert nil alignment.
      alignment->Map(s.second, Handle::nil());
    } else {
      // Matching span found. Add mention pair to alignment.
      alignment->Map(s.second, f->second);
    }
  }
}

void FrameEvaluation::AlignEvokes(Store *store,
                                  const Alignment &mentions,
                                  Alignment *alignment) {
  Handle n_evokes = store->Lookup("evokes");
  for (const auto &m : mentions) {
    if (m.second != Handle::nil()) {
      // Align source and target mentions.
      Frame source(store, m.first);
      Frame target(store, m.second);
      AlignEvoke(source, target, n_evokes, alignment);
    } else {
      // Add empty alignments for all frames evoked by the source.
      Frame source(store, m.first);
      for (const Slot &s : source) {
        if (s.name == n_evokes) {
          alignment->Map(s.value, Handle::nil());
        }
      }
    }
  }
}

void FrameEvaluation::AlignEvoke(const Frame &source,
                                 const Frame &target,
                                 Handle n_evokes,
                                 Alignment *alignment) {
  int source_evokes = SlotCount(source, n_evokes);
  int target_evokes = SlotCount(target, n_evokes);
  if (source_evokes == 1 && target_evokes == 1) {
    // Each span only evokes a single frame.
    alignment->Map(source.GetHandle(n_evokes), target.GetHandle(n_evokes));
  } else if (source_evokes > 0 && target_evokes > 0) {
    // Align evoked frames based on type.
    for (const Slot &s : source) {
      if (s.name != n_evokes) continue;

      // Get type for frame evoked by source.
      Frame source_frame(source.store(), s.value);
      Handle source_type = source_frame.GetHandle(Handle::isa());
      if (source_type.IsNil()) {
        alignment->Map(source_frame.handle(), Handle::nil());
        continue;
      }

      // Try to find frame evoked by target with same type.
      Handle match = Handle::nil();
      for (const Slot &t : target) {
        if (t.name != n_evokes) continue;
        Frame target_frame(target.store(), t.value);
        Handle target_type = target_frame.GetHandle(Handle::isa());
        if (target_type == source_type) {
          match = target_frame.handle();
          break;
        }
      }

      // Add alignment for frame evoked from source mention. This will be nil
      // if no match was found. This ensures that all frames evoked from
      // mentions will have an entry in the alignment.
      alignment->Map(source_frame.handle(), match);
    }
  } else if (source_evokes > 0) {
    // Add empty alignment for all source frames.
    for (const Slot &s : source) {
      if (s.name == n_evokes) {
        alignment->Map(s.value, Handle::nil());
      }
    }
  }
}

void FrameEvaluation::AlignFrames(Store *store, Alignment *alignment) {
  // Initialize queue of all the frame pairs where the slots still need to be
  // aligned.
  std::vector<FramePair> pending;
  for (auto it : *alignment) {
    if (!it.second.IsNil()) pending.push_back(it);
  }

  // Keep aligning the slots in the frame pairs in the pending queue.
  while (!pending.empty()) {
    // Get next pending frame from the queue.
    FramePair current = pending.back();
    pending.pop_back();
    Frame source(store, current.first);
    Frame target(store, current.second);

    // Try to find alignment for each slot in the source frame.
    for (const Slot &s : source) {
      // Skip special slots.
      if (s.name.IsId() || s.name.IsIsA() || s.name.IsIs()) continue;

      // Skip slots that do no refer to a local frame. These are typically
      // labels and not frame-to-frame roles.
      if (!s.value.IsLocalRef()) continue;
      Frame value(store, s.value);
      if (!value.IsFrame()) continue;

      // Skip if already aligned.
      if (!alignment->Lookup(value.handle()).IsNil()) continue;

      // Find corresponding role value in target.
      Handle h = target.GetHandle(s.name);

      // Add alignment for role value. An entry is added even in the case
      // where there is no target to ensure that all source frames will have
      // an entry in the alignment.
      alignment->Map(value.handle(), h);

      // Add frame pair to the pending frame alignment queue.
      if (!h.IsNil()) pending.emplace_back(value.handle(), h);
    }
  }
}

void FrameEvaluation::AlignmentAccuracy(
    const Alignment &alignment, Metric *metric) {
  for (const auto &a : alignment) {
    metric->prediction(!a.second.IsNil());
  }
}

void FrameEvaluation::RoleAccuracy(
    Store *store, const Alignment &alignment,
    Metric *pair, Metric *edge, Metric *role,
    Metric *type, Metric *label) {
  for (const auto &a : alignment) {
    Frame source(store, a.first);
    Frame target(store, a.second);

    // Try to find match target slot for each slot in the source frame.
    for (const Slot &s : source) {
      if (s.name.IsIsA()) {
        // Check type.
        type->prediction(HasSlot(target, Handle::isa(), s.value));
      } else if (s.name.IsId() || s.name.IsIs()) {
        // Ignore special roles.
      } else if (s.value.IsLocalRef()) {
        // Check frame-to-frame role.
        Handle value = alignment.Lookup(s.value);
        pair->prediction(!value.IsNil());
        edge->prediction(HasValue(target, value));
        role->prediction(HasSlot(target, s.name, value));
      } else {
        // Check label role.
        label->prediction(HasSlot(target, s.name, s.value));
      }
    }
  }
}

int FrameEvaluation::SlotCount(const Frame &f, Handle name) {
  int n = 0;
  for (const Slot &s : f) {
    if (s.name == name) n++;
  }
  return n;
}

bool FrameEvaluation::HasSlot(const Frame &f, Handle name, Handle value) {
  if (f.invalid() || name.IsNil() || value.IsNil()) return false;
  for (const Slot &s : f) {
    if (s.name == name && s.value == value) return true;
  }
  return false;
}

bool FrameEvaluation::HasValue(const Frame &f, Handle value) {
  if (f.invalid() || value.IsNil()) return false;
  for (const Slot &s : f) {
    if (s.value == value) return true;
  }
  return false;
}

}  // namespace nlp
}  // namespace sling
