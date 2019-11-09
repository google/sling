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

# Script for training a Sempar model from scratch.
#
# It should be run from the top-level folder (i.e. the one which contains the
# 'sling' subfolder).
#
# Usage:
#   /path/to/this/script <flags described in train_pytorch.py and train_util.py>
#
# Sample command:
#   /path/to/this/script [--commons=commons_file] --train=train.rec --dev=dev.rec \
#        --output=out_folder [--word_embeddings=word_embeddings.bin] \
#        [--batch=8] [--report_every=3000] [--steps=100000] [--alpha=0.005]
#
# It takes as input:
# - A file pattern of training documents with gold annotations. Each file should
#   correspond to one serialized Document frame.
# - Optional: File path to the commons store.
#   If no path is specified or the path doesn't exist, then a commons store
#   will be automatically created.
# - Name of the output folder. All generated resources e.g. feature lexicons,
#   action table, master spec, TF graph, trained model etc. will be dumped here.
# - Word embedding dimension, and optionally a pretrained word embedding.
# - Training parameters, e.g. batch size, training steps, learning rate,
#   checkpoint interval etc.
#
# It outputs a Myelin flow file in the output folder corresponding to the best
# checkpoint.


set -eu

readonly COMMAND=`echo $0 $@`
readonly ARGS=`echo $@`

OUTPUT_FOLDER=
for i in "$@"
do
case $i in
    --output_dir=*|--output=*|--output_folder=*)
    OUTPUT_FOLDER="${i#*=}"
    shift
    ;;
    *)
    shift
    ;;
esac
done

if [ -z "$OUTPUT_FOLDER" ];
then
  echo "Output folder not specified. Use --output_folder to specify it."
  exit 1
fi

mkdir -p "${OUTPUT_FOLDER}"

COMMAND_FILE="${OUTPUT_FOLDER}/command"
echo "Writing command to ${COMMAND_FILE}"
echo $COMMAND > ${COMMAND_FILE}

LOG_FILE="${OUTPUT_FOLDER}/log"
echo "Logging will be teed to ${LOG_FILE}"

bazel build -c opt sling/pyapi:pysling.so
SLING_SYMLINK=/usr/local/lib/python2.7/dist-packages/sling
if [ ! -e  $SLING_SYMLINK ];
then
  echo "Need sudo to link the sling python module.."
  sudo ln -s $(realpath python) $SLING_SYMLINK
fi
stdbuf -o 0 python3 sling/nlp/parser/tools/train_pytorch.py $ARGS 2>&1 | tee ${LOG_FILE}
echo "Done. Log is available at ${LOG_FILE}."
