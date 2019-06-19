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

TESTPGM="python3 sling/myelin/tests/opcheck.py"
EXTRA=$@

# Determine CPU feature support.
AVX512=$(grep avx512 /proc/cpuinfo)
FMA=$(grep fma /proc/cpuinfo)
AVX2=$(grep avx2 /proc/cpuinfo)
AVX=$(grep avx /proc/cpuinfo)

# Run all CPU tests for data type.
testcpu() {
  DT=$1
  echo "Test data type $DT"
  $TESTPGM --dt $DT ${EXTRA}

  if [[ $AVX512 ]]; then
    echo "Test data type $DT without AVX512"
    $TESTPGM --dt $DT --cpu=-avx512 ${EXTRA}
  fi
  if [[ $FMA ]]; then
    echo "Test data type $DT without FMA3"
    $TESTPGM --dt $DT --cpu=-avx512-fma3 ${EXTRA}
  fi
  if [[ $AVX2 ]]; then
    echo "Test data type $DT without AVX2"
    $TESTPGM --dt $DT --cpu=-avx512-avx2 ${EXTRA}
    if [[ $FMA ]]; then
      echo "Test data type $DT without AVX2 and FMA3"
      $TESTPGM --dt $DT --cpu=-avx512-fma3-avx2 ${EXTRA}
    fi
  fi
  if [[ $AVX ]]; then
    echo "Test data type $DT without AVX"
    $TESTPGM --dt $DT --cpu=-avx512-fma3-avx2-avx ${EXTRA}
  fi
}

# Run all GPU tests for data type.
testgpu() {
  DT=$1
  echo "Test data type $DT on GPU"
  $TESTPGM --gpu --dt $DT ${EXTRA}
}

# Stop on errors.
set -e

# Test float types on CPU.
testcpu float32
testcpu float64

# Test integer types on CPU.
testcpu int8
testcpu int16
testcpu int32
testcpu int64

# Test on GPU if CUDA is installed.
if [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
  # Test float types on GPU.
  testgpu float32
  testgpu float64

  # Test integer types on GPU.
  testgpu int16
  testgpu int32
  testgpu int64
fi

echo "==== ALL TESTS PASSED ====="

