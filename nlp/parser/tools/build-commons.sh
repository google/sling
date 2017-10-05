#!/bin/bash
#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script for building commons store for OntoNotes sempar.

# Build binaries.
bazel build -c opt nlp/parser/tools:build-store

# Convert PropBank to SLING format.
python nlp/parser/tools/propbank-to-sling.py

# Build commons.
bazel-bin/nlp/parser/tools/build-store -o /tmp/commons \
  data/nlp/schemas/meta-schema.sling \
  data/nlp/schemas/document-schema.sling \
  data/nlp/schemas/propbank-schema.sling \
  data/nlp/schemas/saft-schema.sling \
  data/nlp/schemas/catalog.sling \
  /tmp/propbank.sling \

