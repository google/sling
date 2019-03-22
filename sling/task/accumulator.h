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

#ifndef SLING_TASK_ACCUMULATOR_H_
#define SLING_TASK_ACCUMULATOR_H_

#include <string>
#include <vector>

#include "sling/base/types.h"
#include "sling/task/message.h"
#include "sling/task/reducer.h"
#include "sling/task/task.h"
#include "sling/string/text.h"
#include "sling/util/mutex.h"

namespace sling {
namespace task {

// Accumulator for collecting counts for keys.
class Accumulator {
 public:
  // Initialize accumulator.
  void Init(Channel *output, int num_buckets = 1 << 20);

  // Add counts for string key.
  void Increment(Text key, int64 count = 1);

  // Add counts for numeric key.
  void Increment(uint64 key, int64 count = 1);

  // Flush remaining counts to output.
  void Flush();

 private:
  // Hash buckets for accumulating counts.
  struct Bucket {
    string key;
    uint64 hash = 0;
    int64 count = 0;
  };
  std::vector<Bucket> buckets_;

  // Output channel for accumulated counts.
  Channel *output_ = nullptr;

  // Statistics.
  Counter *num_slots_used_ = nullptr;
  Counter *num_collisions_ = nullptr;

  // Mutex for serializing access to accumulator.
  Mutex mu_;
};

// Reducer that outputs the sum of all the values for a key.
class SumReducer : public Reducer {
 public:
  // Initialize reducer.
  void Start(Task *task) override;

  // Sum all the counts for the key and call the output method with the sum.
  void Reduce(const ReduceInput &input) override;

  // Called with aggregate count for key. The default implementation just
  // outputs the key and the sum to the output.
  virtual void Aggregate(int shard, const Slice &key, uint64 sum);

 private:
  // Discard keys with counts lower than the threshold.
  int64 threshold_ = 0;

  // Statistics.
  Counter *num_keys_discarded_ = nullptr;
  Counter *num_counts_discarded_ = nullptr;
};

}  // namespace task
}  // namespace sling

#endif  // SLING_TASK_ACCUMULATOR_H_

