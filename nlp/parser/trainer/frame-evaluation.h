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

// Frame evaluation.

#ifndef NLP_PARSER_TRAINER_FRAME_EVALUATION_H_
#define NLP_PARSER_TRAINER_FRAME_EVALUATION_H_

#include <hash_map>
#include <string>
#include <utility>
#include <vector>

#include "nlp/document/document.h"
#include "frame/object.h"
#include "frame/store.h"
#include "util/hash.h"

namespace sling {
namespace nlp {

// Compute precision and recall for frame annotations in an annotated corpus
// compared to a gold-standard corpus. This evaluation does not take thematic
// frames into account yet.
class FrameEvaluation {
 public:
  // Span with begin and end.
  typedef std::pair<int, int> Span;

  // Pair of frames.
  typedef std::pair<Handle, Handle> FramePair;

  // Hasher for Span.
  struct SpanHash {
    size_t operator()(const Span &span) const {
      return (span.first << 20) | (span.second - span.first);
    }
  };

  // Mapping from span to mention frame.
  typedef hash_map<Span, Handle, SpanHash> MentionMap;

  // Frame alignment.
  class Alignment : public HandleMap<Handle> {
   public:
    // Maps source frame to target frame. Returns true if the mapping was added
    // to the alignment.
    bool Map(Handle source, Handle target);

    // Returns the target that frame is mapped to or nil.
    Handle Lookup(Handle handle) const;

   private:
    HandleSet targets_;
  };

  // Statistics for computing accuracy for one metric.
  struct Metric {
    // Adds one correct/wrong prediction to metric.
    void prediction(bool good) {
      if (good) {
        correct++;
      } else {
        wrong++;
      }
    }

    // Adds another metric to this one.
    void add(const Metric &other) {
      correct += other.correct;
      wrong += other.wrong;
    }

    // Total number of predictions.
    int total() const {
      return correct + wrong;
    }

    // Prediction accuracy.
    double accuracy() const {
      if (total() == 0) return 0;
      return static_cast<double>(correct) / static_cast<double>(total());
    }

    // Number of correct and wrong predictions.
    int correct = 0;
    int wrong = 0;
  };

  // Benchmark with precision and recall.
  struct Benchmark {
    // Computes F-score from precision and recall.
    double fscore() const {
      double p = precision.accuracy();
      double r = recall.accuracy();
      if (p == 0 && r == 0) return 0;
      return 2 * p * r / (p + r);
    }

    // Adds another benchmark to this one.
    void add(const Benchmark other) {
      recall.add(other.recall);
      precision.add(other.precision);
    }

    // Precision and recall statistics for benchmark.
    Metric recall;
    Metric precision;

    // Textualizes benchmark in 'lines'.
    void ToText(const string &name, std::vector<string> *lines) const;
  };

  // Holds evaluation output.
  struct Output {
    // Benchmarks of various aspects.
    Benchmark mention;
    Benchmark frame;
    Benchmark type;
    Benchmark role;
    Benchmark label;
    Benchmark slot;
    Benchmark combined;

    // Counters.
    int64 num_golden_spans = 0;
    int64 num_predicted_spans = 0;
    int64 num_golden_frames = 0;
    int64 num_predicted_frames = 0;
  };

  // Evaluates two equal-sized corpora of files (gold and test) and returns
  // the evaluation in 'output'.
  static void Evaluate(Store *commons,
                       const string &gold_file_pattern,
                       const string &test_file_pattern,
                       Output *output);

  // Same as above, except it reads the commons from 'commons_file', and
  // saves the final evaluation in the file with path 'output_file'.
  static void EvaluateAndWrite(const string &commons_file,
                               const string &gold_file_pattern,
                               const string &test_file_pattern,
                               const string &output_file);

 private:
  // Constructs mention map from spans to mention frames.
  static void GetMentionMap(const Frame &frame, MentionMap *mentions);

  // Computes mention alignment from source to target.
  static void AlignMentions(const MentionMap &source,
                            const MentionMap &target,
                            Alignment *alignment);

  // Computes alignment between evoked frames for each mention.
  static void AlignEvokes(Store *store,
                          const Alignment &mentions,
                          Alignment *alignment);

  // Align evoked frames in mention source with evoked frames in mention target.
  static void AlignEvoke(const Frame &source,
                         const Frame &target,
                         Handle n_evokes,
                         Alignment *alignment);

  // Extends frame alignment to all remaining frames reachable from the initial
  // alignment with the evoked frames.
  static void AlignFrames(Store *store, Alignment *alignment);

  // Computes alignment accuracy.
  static void AlignmentAccuracy(const Alignment &alignment, Metric *metric);

  // Computes role accuracy.
  static void RoleAccuracy(Store *store, const Alignment &alignment,
                           Metric *type, Metric *role, Metric *label);

  // Counts the number of slots with a given name.
  static int SlotCount(const Frame &f, Handle name);

  // Checks if frame has a role with a given name and value.
  static bool HasRole(const Frame &f, Handle name, Handle value);
};

}  // namespace nlp
}  // namespace sling

#endif // NLP_PARSER_TRAINER_FRAME_EVALUATION_H_
