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

#include "nlp/parser/trainer/frame-evaluation.h"

#include <algorithm>

#include "base/logging.h"
#include "frame/serialization.h"
#include "string/strcat.h"

namespace sling {
namespace nlp {

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

namespace {

Document *ReadDocumentOrDie(Store *store, const string &file) {
  FileDecoder decoder(store, file);
  Object top = decoder.Decode();
  CHECK(top.valid()) << "Invalid document in " << file;
  CHECK(top.IsFrame()) << "Document not a frame in " << file;
  return new Document(top.AsFrame());
}

}  // namespace

void FrameEvaluation::Evaluate(Store *commons,
                               const string &gold_file_pattern,
                               const string &test_file_pattern,
                               FrameEvaluation::Output *output) {
  // Benchmarks.
  auto &mention = output->mention;
  auto &frame = output->frame;
  auto &type = output->type;
  auto &role = output->role;
  auto &label = output->label;

  // Statistics counters.
  output->num_golden_spans = 0;
  output->num_predicted_spans = 0;
  output->num_golden_frames = 0;
  output->num_predicted_frames = 0;

  // Compare predicted and golden documents.
  std::vector<string> gold_files, test_files;
  CHECK(File::Match(gold_file_pattern, &gold_files));
  CHECK(File::Match(test_file_pattern, &test_files));
  CHECK_EQ(gold_files.size(), test_files.size());

  // Sort the list of filenames to rule out any non-determinism in File::Match.
  // After this we will assume a 1-1 mapping between gold and test documents.
  std::sort(gold_files.begin(), gold_files.end());
  std::sort(test_files.begin(), test_files.end());

  for (int i = 0; i < gold_files.size(); ++i) {
    // Get the next document pair.
    Store store(commons);
    Document *golden = ReadDocumentOrDie(&store, gold_files[i]);
    Document *predicted = ReadDocumentOrDie(&store, test_files[i]);
    CHECK_EQ(golden->num_tokens(), predicted->num_tokens()) << gold_files[i];

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
    AlignEvokes(&store, g2p_mention_alignment, &g2p_frame_alignment);
    AlignEvokes(&store, p2g_mention_alignment, &p2g_frame_alignment);

    // Align frames that are not directly evoked from a span.
    AlignFrames(&store, &g2p_frame_alignment);
    AlignFrames(&store, &p2g_frame_alignment);

    // Compute mention precision and recall.
    AlignmentAccuracy(g2p_mention_alignment, &mention.recall);
    AlignmentAccuracy(p2g_mention_alignment, &mention.precision);

    // Compute frame precision and recall.
    AlignmentAccuracy(g2p_frame_alignment, &frame.recall);
    AlignmentAccuracy(p2g_frame_alignment, &frame.precision);

    // Compute role precision and recall.
    RoleAccuracy(&store, g2p_frame_alignment,
                 &type.recall, &role.recall, &label.recall);
    RoleAccuracy(&store, p2g_frame_alignment,
                 &type.precision, &role.precision, &label.precision);

    // Update statistics.
    output->num_golden_spans += golden_mentions.size();
    output->num_predicted_spans += predicted_mentions.size();
    output->num_golden_frames += g2p_frame_alignment.size();
    output->num_predicted_frames += p2g_frame_alignment.size();

    delete golden;
    delete predicted;
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
  std::vector<string> lines;
  eval.mention.ToText("SPAN", &lines);
  eval.frame.ToText("FRAME", &lines);
  eval.type.ToText("TYPE", &lines);
  eval.role.ToText("ROLE", &lines);
  eval.label.ToText("LABEL", &lines);
  eval.slot.ToText("SLOT", &lines);
  eval.combined.ToText("COMBINED", &lines);
  lines.emplace_back(StrCat("#GOLDEN_SPANS\t", eval.num_golden_spans));
  lines.emplace_back(StrCat("#PREDICTED_SPANS\t", eval.num_predicted_spans));
  lines.emplace_back(StrCat("#GOLDEN_FRAMES\t", eval.num_golden_frames));
  lines.emplace_back(StrCat("#PREDICTED_FRAMES\t", eval.num_predicted_frames));

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

void FrameEvaluation::Benchmark::ToText(const string &name,
                                        std::vector<string> *lines) const {
  double p = precision.accuracy();
  double r = recall.accuracy();
  double f1 = fscore();
  lines->emplace_back(StrCat(name, "_P+", "\t", precision.correct));
  lines->emplace_back(StrCat(name, "_P-", "\t", precision.wrong));
  lines->emplace_back(StrCat(name, "_R+", "\t", recall.correct));
  lines->emplace_back(StrCat(name, "_R-", "\t", recall.wrong));
  lines->emplace_back(StrCat(name, "_Precision", "\t", p * 100.0));
  lines->emplace_back(StrCat(name, "_Recall", "\t", r * 100.0));
  lines->emplace_back(StrCat(name, "_F1", "\t", f1 * 100.0));
}

void FrameEvaluation::GetMentionMap(
    const Frame &frame, MentionMap *mentions) {
  Store *store = frame.store();
  Handle n_mention = store->Lookup("/s/document/mention");
  Handle n_begin = store->Lookup("/s/phrase/begin");
  Handle n_length = store->Lookup("/s/phrase/length");

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
  Handle n_evokes = store->Lookup("/s/phrase/evokes");
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
    Metric *type, Metric *role, Metric *label) {
  for (const auto &a : alignment) {
    Frame source(store, a.first);
    Frame target(store, a.second);

    // Try to find match target slot for each slot in the source frame.
    for (const Slot &s : source) {
      if (s.name.IsIsA()) {
        // Check type.
        type->prediction(HasRole(target, Handle::isa(), s.value));
      } else if (s.name.IsId() || s.name.IsIs()) {
        // Ignore special roles.
      } else if (s.value.IsLocalRef()) {
        // Check frame-to-frame role.
        role->prediction(HasRole(target, s.name, alignment.Lookup(s.value)));
      } else {
        // Check label role.
        label->prediction(HasRole(target, s.name, s.value));
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

bool FrameEvaluation::HasRole(const Frame &f, Handle name, Handle value) {
  if (f.invalid() || name.IsNil() || value.IsNil()) return false;
  for (const Slot &s : f) {
    if (s.name == name && s.value == value) return true;
  }
  return false;
}

}  // namespace nlp
}  // namespace sling
