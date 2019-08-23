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
  num_slots_used_ = task->GetCounter("accumulator_slots_used");
  num_collisions_ = task->GetCounter("accumulator_collisions");
}

void Accumulator::Increment(Text key, int64 count) {
  uint64 fp = Fingerprint(key.data(), key.size());
  uint32 b = (fp ^ (fp >> 32)) % buckets_.size();
  MutexLock lock(&mu_);
  Bucket &bucket = buckets_[b];
  if (fp != bucket.hash || key != bucket.key) {
    if (bucket.count != 0) {
      output_->Send(new Message(bucket.key, SimpleItoa(bucket.count)));
      bucket.count = 0;
      num_collisions_->Increment();
    } else {
      num_slots_used_->Increment();
    }
    bucket.hash = fp;
    bucket.key.assign(key.data(), key.size());

  }
  bucket.count += count;
}

void Accumulator::Increment(uint64 key, int64 count) {
  uint64 b = (key ^ (key >> 32)) % buckets_.size();
  MutexLock lock(&mu_);
  Bucket &bucket = buckets_[b];
  if (key != bucket.hash) {
    if (bucket.count != 0) {
      output_->Send(new Message(bucket.key, SimpleItoa(bucket.count)));
      bucket.count = 0;
      num_collisions_->Increment();
    } else {
      num_slots_used_->Increment();
    }
    bucket.hash = key;
    bucket.key = SimpleItoa(key);
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

void SumReducer::Start(Task *task) {
  Reducer::Start(task);
  task->Fetch("threshold", &threshold_);
  if (threshold_ > 0) {
    num_keys_discarded_ = task->GetCounter("keys_discarded");
    num_counts_discarded_ = task->GetCounter("counts_discarded");
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
  if (sum >= threshold_) {
    Aggregate(input.shard(), input.key(), sum);
  } else {
    if (num_keys_discarded_) num_keys_discarded_->Increment();
    if (num_counts_discarded_) num_counts_discarded_->Increment(sum);
  }
}

void SumReducer::Aggregate(int shard, const Slice &key, uint64 sum) {
  Output(shard, new Message(key, SimpleItoa(sum)));
}

REGISTER_TASK_PROCESSOR("sum-reducer", SumReducer);

}  // namespace task
}  // namespace sling

