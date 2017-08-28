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

#include <iostream>
#include <string>
#include "base/init.h"

// Wrapper around InitProgram that is automatically called when an object is
// constructed. One such object is the global variable below.
class MyInit {
 public:
  MyInit() {
    int argc = 0;
    sling::InitProgram(&argc, NULL);
  }
};

// Calls MyInit() automatically.
MyInit _my_init_;
