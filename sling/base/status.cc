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

#include "sling/base/status.h"

#include <stdlib.h>
#include <string.h>
#include <string>

#include "sling/base/logging.h"
#include "sling/base/types.h"

namespace sling {

const Status &Status::OK = Status();

Status::State *Status::CopyState(const State *s) {
  if (s == nullptr) return nullptr;
  int size = sizeof(State) + s->length + 1;
  State *result = static_cast<State *>(malloc(size));
  CHECK(result != nullptr);
  memcpy(result, s, size);
  return result;
}

Status::Status(int code, const char *msg) {
  DCHECK_NE(code, 0);
  int length = strlen(msg);
  int size = sizeof(State) + length + 1;
  state_ = static_cast<State *>(malloc(size));
  CHECK(state_ != nullptr);
  state_->length = length;
  state_->code = code;
  memcpy(state_->message(), msg, length + 1);
}

Status::Status(int code, const char *msg1, const char *msg2)  {
  DCHECK_NE(code, 0);
  int length1 = strlen(msg1);
  int length2 = strlen(msg2);
  int length = length1 + length2 + 2;
  int size = sizeof(State) + length + 1;
  state_ = static_cast<State *>(malloc(size));
  CHECK(state_ != nullptr);
  state_->length = length;
  state_->code = code;
  char *msg = state_->message();
  memcpy(msg, msg1, length1);
  msg[length1] = ':';
  msg[length1 + 1] = ' ';
  memcpy(msg + length1 + 2, msg2, length2 + 1);
}

Status::Status(int code, const char *msg1, const string &msg2)
  : Status(code, msg1, msg2.c_str()) {}

Status::Status(int code, const string &msg)
  : Status(code, msg.c_str()) {}

string Status::ToString() const {
  if (state_ == nullptr) {
    return "OK";
  } else {
    return "ERROR " + std::to_string(state_->code) + " : " + state_->message();
  }
}

}  // namespace sling

