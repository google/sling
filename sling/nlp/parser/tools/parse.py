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

# Script for running inference/evaluation with a flow file.
# Usage:
# python sling/nlp/parser/tools/parse.py \
#   --parser=<path to flow file> \
#   --input=<input recordio> \
#   --output=<output recordio> \
#   [--evaluate]

import os
import sling
import sling.flags as flags
from sling.myelin.flow import Flow
import sys
import tempfile

sys.path.insert(0, "sling/nlp/parser/trainer")
from corpora import Corpora
from pytorch_modules import Caspar
from spec import Spec
from train_util import *

def run(args):
  check_present(args, ["input", "parser", "output"])
  assert os.path.exists(args.input), args.input
  assert os.path.exists(args.parser), args.parser

  # Read parser flow.
  flow = Flow()
  flow.load(args.parser)

  # Initialize the spec from the flow.
  spec = Spec()
  spec.from_flow(flow)

  # Initialize the model from the flow.
  caspar = Caspar(spec)
  caspar.from_flow(flow)

  corpus = Corpora(args.input, caspar.spec.commons)
  writer = sling.RecordWriter(args.output)
  count = 0
  for document in corpus:
    state, _, _, trace = caspar.forward(document, train=False, debug=args.trace)
    state.write()
    if trace:
      trace.write()
    writer.write(str(count), state.encoded())
    count += 1
    if count % 100 == 0:
      print("Annotated", count, "documents", now(), mem())
  writer.close()
  print("Annotated", count, "documents", now(), mem())
  print("Wrote annotated documents to", args.output)

  if args.evaluate:
    f = tempfile.NamedTemporaryFile(delete=False)
    fname = f.name
    caspar.spec.commons.save(fname, binary=True)
    f.close()
    eval_result = frame_evaluation(gold_corpus_path=args.input, \
        test_corpus_path=args.output, commons=caspar.spec.commons)
    os.unlink(fname)
    return eval_result


if __name__ == '__main__':
  setup_runtime_flags(flags)
  flags.parse()
  run(flags.arg)
