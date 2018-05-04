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

#include "sling/base/logging.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string>

#include "sling/base/macros.h"
#include "sling/base/types.h"

namespace sling {

int LogMessage::loglevel = 0;
int LogMessage::vloglevel = 0;

LogMessage::LogMessage(const char *fname, int line, int severity)
    : fname_(fname), line_(line), severity_(severity) {}

void LogMessage::GenerateLogMessage() {
  const size_t BUFSIZE = 30;
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  char timestr[BUFSIZE];
  strftime(timestr, BUFSIZE, "%Y-%m-%d %H:%M:%S", localtime(&tv.tv_sec));

  // TODO: Replace this with something that logs through a log sink.
  fprintf(stderr, "[%s.%06ld: %c %s:%d] %s\n", timestr, tv.tv_usec,
          "IWEF"[severity_], fname_, line_, str().c_str());
}

LogMessage::~LogMessage() {
  if (severity_ >= loglevel) GenerateLogMessage();
}

LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, FATAL) {}

LogMessageFatal::~LogMessageFatal() {
  GenerateLogMessage();
  abort();
}

template <>
void MakeCheckOpValueString(std::ostream *os, const char &v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "char value " << static_cast<short>(v);
  }
}

template <>
void MakeCheckOpValueString(std::ostream *os, const signed char &v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "signed char value " << static_cast<short>(v);
  }
}

template <>
void MakeCheckOpValueString(std::ostream *os, const unsigned char &v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "unsigned char value " << static_cast<unsigned short>(v);
  }
}

template <>
void MakeCheckOpValueString(std::ostream *os, const std::nullptr_t &p) {
  (*os) << "nullptr";
}

CheckOpMessageBuilder::CheckOpMessageBuilder(const char *exprtext)
    : stream_(new std::ostringstream) {
  *stream_ << "Check failed: " << exprtext << " (";
}

CheckOpMessageBuilder::~CheckOpMessageBuilder() {
  delete stream_;
}

std::ostream* CheckOpMessageBuilder::ForVar2() {
  *stream_ << " vs. ";
  return stream_;
}

string *CheckOpMessageBuilder::NewString() {
  *stream_ << ") ";
  return new string(stream_->str());
}

}  // namespace sling

