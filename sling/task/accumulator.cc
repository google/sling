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

#include "sling/task/accumulator.h"

#include "sling/base/logging.h"
#include "sling/string/numbers.h"
#include "sling/task/reducer.h"
#include "sling/util/fingerprint.h"
#include "sling/util/mutex.h"

namespace sling {
namespace task {

void Accumulator::Init(Channel *output, int num_buckets) {
  output_ = output;
  buckets_.clear();
  buckets_.resize(num_buckets);

  Task *task = output->producer().task();
  num_slots_used_ = task->GetCounter("num_accumulator_slots_used");
  num_collisions_ = task->GetCounter("num_accumulator_collisions");
}

void Accumulator::Increment(Text key, int64 count) {
  uint64 b = Fingerprint(key.data(), key.size()) % buckets_.size();
  MutexLock lock(&mu_);
  Bucket &bucket = buckets_[b];
  if (key != bucket.key) {
    if (bucket.count != 0) {
      output_->Send(new Message(bucket.key, SimpleItoa(bucket.count)));
      bucket.count = 0;
      num_collisions_->Increment();
    } else {
      num_slots_used_->Increment();
    }
    bucket.key.assign(key.data(), key.size());
  }
  bucket.count += count;
}

void Accumulator::Flush() {
  MutexLock lock(&mu_);
  for (Bucket &bucket : buckets_) {
    if (bucket.count != 0) {
      output_->Send(new Message(bucket.key, SimpleItoa(bucket.count)));
      bucket.count = 0;
    }
    bucket.key.clear();
  }
}

void SumReducer::Reduce(const ReduceInput &input) {
  int64 sum = 0;
  for (Message *m : input.messages()) {
    int64 count;
    const Slice &value = m->value();
    CHECK(safe_strto64_base(value.data(), value.size(), &count, 10));
    sum += count;
  }
  Aggregate(input.shard(), input.key(), sum);
}

void SumReducer::Aggregate(int shard, const Slice &key, uint64 sum) {
  Output(shard, new Message(key, SimpleItoa(sum)));
}

REGISTER_TASK_PROCESSOR("sum-reducer", SumReducer);

}  // namespace task
}  // namespace sling

