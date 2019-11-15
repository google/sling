#/bin/bash
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

# Script for downloading CoNLL OntoNotes 5 data and converting it to a SLING
# corpus that can be used for training and testing a SLING parser.
#
# The LDC2013T19 OntoNotes 5 corpus is needed for the conversion. This is
# licensed by LDC and you need an LDC license to get the corpus:
#   https://catalog.ldc.upenn.edu/LDC2013T19
#
# LDC2013T19.tar.gz is assumed to be in local/data/corpora/ontonotes.
#
# The OntoNotes SLING corpus will end up in local/data/corpora/caspar.

set -e

ONTONOTES=local/data/corpora/ontonotes
pushd $ONTONOTES

echo "Check that OntoNotes 5 corpus is present"
if [ -f "LDC2013T19.tar.gz" ] ; then
  echo "OntoNotes 5 corpus present"
else
  echo "OntoNotes 5 corpus not found"
  echo "OntoNotes 5 can be obtained from LDC if you have a LDC license"
  echo "See: https://catalog.ldc.upenn.edu/LDC2013T19"
  exit 1
fi

echo "Unpack OntoNotes 5"
tar -xf LDC2013T19.tar.gz

echo "Download and unpack the CoNLL formated OntoNotes 5 data"
wget https://github.com/ontonotes/conll-formatted-ontonotes-5.0/archive/v12.tar.gz
tar -xf v12.tar.gz --strip-components=1

wget -O train.ids http://conll.cemantix.org/2012/download/ids/english/coref/train.id
wget -O dev.ids http://conll.cemantix.org/2012/download/ids/english/coref/development.id
wget -O test.ids http://conll.cemantix.org/2012/download/ids/english/coref/test.id

wget http://ontonotes.cemantix.org/download/conll-formatted-ontonotes-5.0-scripts.tar.gz
tar -xf conll-formatted-ontonotes-5.0-scripts.tar.gz

echo "Generate CoNLL files"
./conll-formatted-ontonotes-5.0/scripts/skeleton2conll.sh -D ontonotes-release-5.0/data/files/data/ conll-formatted-ontonotes-5.0/

popd

echo "Convert CoNLL files to SLING"

CONVERTER=sling/nlp/parser/ontonotes/ontonotesv5_to_sling.py
SHUFFLE=sling/nlp/parser/ontonotes/shuffle.py
IN=$ONTONOTES/conll-formatted-ontonotes-5.0/data
OUT=local/data/corpora/caspar

mkdir -p $OUT

python3 $CONVERTER \
  --input=$IN/train/data/english/annotations/ \
  --allowed_ids=$ONTONOTES/train.ids \
  --output=$OUT/train.rec

python3 $CONVERTER \
  --input=$IN/development/data/english/annotations/ \
  --allowed_ids=$ONTONOTES/dev.ids \
  --output=$OUT/dev.rec

python3 $CONVERTER \
  --input=$IN/test/data/english/annotations/ \
  --allowed_ids=$ONTONOTES/test.ids \
  --output=$OUT/test.rec

echo "Shuffle training corpus"

python3 $SHUFFLE --input $OUT/train.rec --output $OUT/train_shuffled.rec

echo "Done."
