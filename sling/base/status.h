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

#ifndef SLING_BASE_STATUS_H_
#define SLING_BASE_STATUS_H_

#include <stdlib.h>
#include <ostream>
#include <string>

#include "sling/base/logging.h"
#include "sling/base/types.h"

namespace sling {

// A Status encapsulates the result of an operation.  It may indicate success,
// or it may indicate an error with an associated error message.
class Status {
 public:
  // Create a success status.
  Status() : state_(nullptr) { }
  ~Status() { free(state_); }

  // Create an error status.
  Status(int code, const char *msg);
  Status(int code, const char *msg1, const char *msg2);
  Status(int code, const string &msg);
  Status(int code, const char *msg1, const string &msg2);

  // Copy the specified status.
  Status(const Status &s) : state_(CopyState(s.state_)) {}

  void operator=(const Status &s) {
    if (state_ != s.state_) {
      free(state_);
      state_ = CopyState(s.state_);
    }
  }

  // Status comparison.
  bool operator==(const Status &s) const { return s.code() == code(); }
  bool operator!=(const Status &s) const { return s.code() != code(); }

  // Returns true iff the status indicates success.
  bool ok() const { return state_ == nullptr; }

  // This bool operator returns true if status is ok. If status is not ok,
  // it will also log an error message. This can be used for checking for
  // errors with the CHECK macro, e.g. CHECK(File::MkDir(...));
  operator bool() const {
    if (!ok()) LOG(ERROR) << ToString();
    return ok();
  }

  // Return a string representation of this status suitable for printing.
  // Returns the string "OK" for success.
  string ToString() const;

  // Returns the error code or zero for success.
  int code() const { return state_ == nullptr ? 0 : state_->code; }

  // Returns error message or empty string for success.
  const char *message() const {
    return state_ == nullptr ? "" : state_->message();
  }

  // Success status.
  static const Status &OK;

 private:
  // If status is OK, state is null.  Otherwise, state points to a State
  // struct with the following information:
  struct State {
    int length;  // length of error message
    int code;    // error code.
    // followed by the error message (null terminated).
    char *message() { return reinterpret_cast<char *>(this + 1); }
  };

  State *state_;

  // Clone state.
  static State *CopyState(const State *s);
};

// Output status to stream.
inline std::ostream &operator<<(std::ostream &out, const Status &status) {
  out << status.ToString();
  return out;
}

}  // namespace sling

#endif  // SLING_BASE_STATUS_H_
