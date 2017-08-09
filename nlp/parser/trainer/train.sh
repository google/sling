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
# 'nlp' and 'frame' subfolders).
# Usage:
#   /path/to/this/script <train filepattern> <path to commons> <output folder>
#
# It takes as input:
# - A file pattern of training documents with gold annotations. Each file should
#   correspond to one serialized Document frame.
# - Path to the commons store.
# - Name of the output folder. All generated resources e.g. feature lexicons,
#   action table, master spec, TF graph, trained model etc. will be dumped here.

# It performs the following steps:
# - Builds the action table.
# - Builds resources needed by the features.
# - Builds a complete MasterSpec proto.
# - Builds a TF graph using the master spec and default hyperparameters.
# - Trains a model using the graph above.

# Tweaks:
# - The features and component attributes (e.g. hidden layer size) are
#   hard-coded in generate-master-spec.cc and can be changed there.
# - The default training hyperparameters are hard-coded in this script and
#   can be changed in $HYPERPARAMS below.

set -eux

SEM=$HOME/sempar_ontonotes
COMMONS=${SEM}/commons
OUTPUT_FOLDER=${SEM}/out
TRAIN_FILEPATTERN=${SEM}/train.zip
DEV_GOLD_FILEPATTERN=${SEM}/dev.gold.zip
DEV_NOGOLD_FILEPATTERN=${SEM}/dev.without-gold.zip
MAKE_SPEC=1
DO_TRAINING=1
BATCH_SIZE=1
REPORT_EVERY=500
LEARNING_RATE=0.0005
TRAIN_STEPS=100000
WORD_EMBEDDINGS_DIM=32
PRETRAINED_WORD_EMBEDDINGS=

for i in "$@"
do
case $i in
    --commons=*)
    COMMONS="${i#*=}"
    shift
    ;;
    --output_dir=*|--output=*|--output_folder=*)
    OUTPUT_FOLDER="${i#*=}"
    shift
    ;;
    --train=*|--train_corpus=*)
    TRAIN_FILEPATTERN="${i#*=}"
    shift
    ;;
    --dev=*|--dev_with_gold=*)
    DEV_GOLD_FILEPATTERN="${i#*=}"
    shift
    ;;
    --dev_without_gold=*)
    DEV_NOGOLD_FILEPATTERN="${i#*=}"
    shift
    ;;
    --spec_only|--only_spec)
    DO_TRAINING=0
    shift
    ;;
    --train_only|--only_train)
    MAKE_SPEC=0
    shift
    ;;
    --batch=*|--batch_size=*)
    BATCH_SIZE="${i#*=}"
    shift
    ;;
    --report_every=*|--checkpoint_every=*)
    REPORT_EVERY="${i#*=}"
    shift
    ;;
    --learning_rate=*|--eta=*)
    LEARNING_RATE="${i#*=}"
    shift
    ;;
    --train_steps=*|--steps=*|--num_train_steps=*)
    TRAIN_STEPS="${i#*=}"
    shift
    ;;
    --word_embeddings_dim=*|--word_dim=*|--word_embedding_dim=*)
    WORD_EMBEDDINGS_DIM="${i#*=}"
    shift
    ;;
    --word_embeddings=*|--pretrained_embeddings=*|--pretrained_word_embeddings=*)
    PRETRAINED_WORD_EMBEDDINGS="${i#*=}"
    shift
    ;;
    *)
    echo "Unknown option " $i
    exit 1
    ;;
esac
done

if [ -z "$COMMONS" ];
then
  echo "Commons not specified. Use --commons to specify it."
  exit 1
fi
if [ -z "$TRAIN_FILEPATTERN" ];
then
  echo "Train corpus not specified. Use --train or --train_corpus."
  exit 1
fi
if [ -z "$DEV_GOLD_FILEPATTERN" ];
then
  echo "Dev gold corpus not specified. Use --dev or --dev_with_gold."
  exit 1
fi
if [ -z "$DEV_NOGOLD_FILEPATTERN" ];
then
  echo "Dev corpus without gold not specified. Use --dev_without_gold."
  exit 1
fi

if [[ "$MAKE_SPEC" -eq 0 ]] && [[ "$DO_TRAINING" -eq 0 ]];
then
  echo "Specify at most one of --only_spec and --only_train"
  exit 1
fi

HYPERPARAMS="learning_rate:${LEARNING_RATE} decay_steps:800000 "
HYPERPARAMS+="seed:2 learning_method:'adam' "
HYPERPARAMS+="use_moving_average:true dropout_rate:1.0 "
HYPERPARAMS+="gradient_clip_norm:1.0 adam_beta1:0.01 "
HYPERPARAMS+="adam_beta2:0.999 adam_eps:0.00001"

if [[ "$MAKE_SPEC" -eq 1 ]];
then
  bazel build -c opt nlp/parser/trainer:generate-master-spec
  bazel-bin/nlp/parser/trainer/generate-master-spec \
    --documents=${TRAIN_FILEPATTERN} \
    --commons=${COMMONS} \
    --output_dir=${OUTPUT_FOLDER} \
    --word_embeddings=${PRETRAINED_WORD_EMBEDDINGS} \
    --word_embeddings_dim=${WORD_EMBEDDINGS_DIM}
fi

if [[ "$DO_TRAINING" -eq 1 ]];
then
  bazel build -c opt nlp/parser/trainer:frame-evaluation
  python nlp/parser/trainer/graph-builder-main.py \
    --master_spec="${OUTPUT_FOLDER}/master_spec" \
    --hyperparams="${HYPERPARAMS}" \
    --output_folder=${OUTPUT_FOLDER} \
    --commons=${COMMONS} \
    --train_corpus=${TRAIN_FILEPATTERN} \
    --dev_corpus=${DEV_GOLD_FILEPATTERN} \
    --dev_corpus_without_gold=${DEV_NOGOLD_FILEPATTERN} \
    --batch_size=${BATCH_SIZE} \
    --report_every=${REPORT_EVERY} \
    --train_steps=${TRAIN_STEPS}
fi

echo "Done."
