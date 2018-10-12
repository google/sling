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

# Script for converting Ontonotes v5 documents to SLING documents.
#
# It should be run from the top-level folder (i.e. the one which contains the
# 'sling' subfolder).
#
# Usage:
#   /path/to/this/script <path to ontonotes directory> <output file.rec> \
#       [optional conversion args]
#
# Example:
# sling/nlp/parser/ontonotes/convert_to_sling.sh \
#   /path/to/ontonotes/data/test/data/english/annotations/  \
#   /tmp/test.rec
#   --allowed_ids_file=<path to ids file>

set -eu

TEMP_RECORDIO=/tmp/output.rec
TEMP_COMMONS=/tmp/commons

echo
echo "Converting Ontonotes to SLING..."
python sling/nlp/parser/ontonotes/ontonotesv5_to_sling.py \
  --input=$1 --output=$TEMP_RECORDIO "${@:3}"

echo
echo "Creating commons..."
python sling/nlp/parser/tools/commons_from_corpora.py \
  --input=$TEMP_RECORDIO --output=$TEMP_COMMONS

echo
echo "Validating converted documents..."
python sling/nlp/parser/tools/validate.py \
  --input=$TEMP_RECORDIO --commons=$TEMP_COMMONS --output=$2

echo
echo "Written converted documents to $2"
echo "Success!"

