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
# - Word embedding dimension, and optionally a pretrained word embedding.
# - Training parameters, e.g. batch size, training steps, learning rate,
#   checkpoint interval etc.

# It performs the following steps:
# - Builds the action table.
# - Builds resources needed by the features.
# - Builds a complete MasterSpec proto.
# - Builds a TF graph using the master spec and default hyperparameters.
# - Trains a model using the graph above.
# - Converts the trained model to a Myelin flow file for use in the runtime.

# Tweaks:
# - The features and component attributes (e.g. hidden layer size) are
#   hard-coded in generate-master-spec.cc and can be changed there.

set -eu

readonly COMMAND=`echo $0 $@`

# Input resources and arguments.
SEM=local/sempar
COMMONS=${SEM}/commons
OUTPUT_FOLDER=${SEM}/out
TRAIN_FILEPATTERN=${SEM}/train.rec
DEV_FILEPATTERN=${SEM}/dev.rec
WORD_EMBEDDINGS_DIM=32
PRETRAINED_WORD_EMBEDDINGS=$SEM/word2vec-32-embeddings.bin
OOV_FEATURES=true

# Training hyperparameters.
BATCH_SIZE=8
REPORT_EVERY=2000
LEARNING_RATE=0.0005
SEED=2
METHOD=adam
ADAM_BETA1=0.01
ADAM_BETA2=0.999
ADAM_EPS=0.00001
GRAD_CLIP_NORM=1.0
DROPOUT=1.0
TRAIN_STEPS=200000
DECAY_STEPS=500000
MOVING_AVERAGE=true

# Whether we should make the MasterSpec again or not.
MAKE_SPEC=1

# Whether we should train or stop after making the MasterSpec.
DO_TRAINING=1

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
    DEV_FILEPATTERN="${i#*=}"
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
    --oov_features=*|--oov_lstm_features=*)
    OOV_FEATURES="${i#*=}"
    shift
    ;;
    --seed=*)
    SEED="${i#*=}"
    shift
    ;;
    --method=*|--optimizer=*)
    METHOD="${i#*=}"
    shift
    ;;
    --adam_beta1=*)
    ADAM_BETA1="${i#*=}"
    shift
    ;;
    --adam_beta2=*)
    ADAM_BETA2="${i#*=}"
    shift
    ;;
    --adam_eps=*|--adam_epsilon=*)
    ADAM_EPS="${i#*=}"
    shift
    ;;
    --grad_clip_norm=*|--gradient_clip_norm=*|--grad_clip=*|--gradient_clip=*)
    GRAD_CLIP_NORM="${i#*=}"
    shift
    ;;
    --dropout=*|--dropout_rate=*|--dropout_keep_rate=*)
    DROPOUT="${i#*=}"
    shift
    ;;
    --decay=*|--decay_steps=*)
    DECAY="${i#*=}"
    shift
    ;;
    --moving_average=*|--use_moving_average=*)
    MOVING_AVERAGE="${i#*=}"
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
if [ -z "$DEV_FILEPATTERN" ];
then
  echo "Dev gold corpus not specified. Use --dev or --dev_with_gold."
  exit 1
fi

if [[ "$MAKE_SPEC" -eq 0 ]] && [[ "$DO_TRAINING" -eq 0 ]];
then
  echo "Specify at most one of --only_spec and --only_train"
  exit 1
fi

HYPERPARAMS="learning_rate:${LEARNING_RATE} decay_steps:${DECAY_STEPS} "
HYPERPARAMS+="seed:${SEED} learning_method:'${METHOD}' "
HYPERPARAMS+="use_moving_average:${MOVING_AVERAGE} dropout_rate:${DROPOUT} "
HYPERPARAMS+="gradient_clip_norm:${GRAD_CLIP_NORM} adam_beta1:${ADAM_BETA1} "
HYPERPARAMS+="adam_beta2:${ADAM_BETA2} adam_eps:${ADAM_EPS}"

mkdir -p "${OUTPUT_FOLDER}"

COMMAND_FILE="${OUTPUT_FOLDER}/command"
echo "Writing command to ${COMMAND_FILE}"
echo $COMMAND > ${COMMAND_FILE}

if [[ "$MAKE_SPEC" -eq 1 ]];
then
  bazel build -c opt sling/nlp/parser/trainer:generate-master-spec
  bazel-bin/sling/nlp/parser/trainer/generate-master-spec \
    --documents=${TRAIN_FILEPATTERN} \
    --commons=${COMMONS} \
    --output_dir=${OUTPUT_FOLDER} \
    --word_embeddings=${PRETRAINED_WORD_EMBEDDINGS} \
    --word_embeddings_dim=${WORD_EMBEDDINGS_DIM} \
    --oov_lstm_features=${OOV_FEATURES} \
    --logtostderr
fi

if [[ "$DO_TRAINING" -eq 1 ]];
then
  bazel build -c opt sling/nlp/parser/tools:evaluate-frames
  bazel build -c opt sling/nlp/parser/trainer:sempar.so
  python sling/nlp/parser/tools/train.py \
    --master_spec="${OUTPUT_FOLDER}/master_spec" \
    --hyperparams="${HYPERPARAMS}" \
    --output_folder=${OUTPUT_FOLDER} \
    --flow=${OUTPUT_FOLDER}/sempar.flow \
    --commons=${COMMONS} \
    --train_corpus=${TRAIN_FILEPATTERN} \
    --dev_corpus=${DEV_FILEPATTERN} \
    --batch_size=${BATCH_SIZE} \
    --report_every=${REPORT_EVERY} \
    --train_steps=${TRAIN_STEPS} \
    --logtostderr
fi

echo "Done."
