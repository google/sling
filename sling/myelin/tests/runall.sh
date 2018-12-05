#!/bin/bash
#
# Copyright 2018 Google Inc.
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

# Run all Myelin tests.

TESTPGM="python sling/myelin/tests/myelin-vs-numpy.py"

# Determine CPU feature support.
AVX512=$(grep avx512 /proc/cpuinfo)
FMA=$(grep fma /proc/cpuinfo)
AVX2=$(grep avx2 /proc/cpuinfo)
AVX=$(grep avx /proc/cpuinfo)

# Run all tests for data type.
testtype() {
  DT=$1
  echo "Test data type $DT"
  $TESTPGM --dt $DT

  if [[ $AVX512 ]]; then
    echo "Test data type $DT without AVX512"
    $TESTPGM --dt $DT --cpu=-avx512
  fi
  if [[ $FMA ]]; then
    echo "Test data type $DT without FMA3"
    $TESTPGM --dt $DT --cpu=-avx512-fma3
  fi
  if [[ $AVX2 ]]; then
    echo "Test data type $DT without AVX2"
    $TESTPGM --dt $DT --cpu=-avx512-avx2
    if [[ $FMA ]]; then
      echo "Test data type $DT without AVX2 and FMA3"
      $TESTPGM --dt $DT --cpu=-avx512-fma3-avx2
    fi
  fi
  if [[ $AVX ]]; then
    echo "Test data type $DT without AVX"
    $TESTPGM --dt $DT --cpu=-avx512-fma3-avx2-avx
  fi
}

# Stop on errors.
set -e

# Test float types.
testtype float32
testtype float64

# Test integer types.
testtype int8
testtype int16
testtype int32
testtype int64

echo "==== ALL TESTS PASSED ====="

