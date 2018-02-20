# SLING - A natural language frame semantics parser

[![Build Status](https://travis-ci.org/google/sling.svg?branch=master)](https://travis-ci.org/google/sling)

SLING is a parser for annotating text with frame semantic annotations. It is
trained on an annotated corpus using [Tensorflow](https://www.tensorflow.org/)
and [Dragnn](https://github.com/tensorflow/models/blob/master/research/syntaxnet/g3doc/DRAGNN.md).

The parser is a general transition-based frame semantic parser using
bi-directional LSTMs for input encoding and a Transition Based Recurrent Unit
(TBRU) for output decoding. It is a jointly trained model using only the text
tokens as input and the transition system has been designed to output frame
graphs directly without any intervening symbolic representation.

![SLING neural network architecture.](./doc/report/network.svg)

The SLING framework includes an efficient and scalable frame store
implementation as well as a neural network JIT compiler for fast parsing at
runtime.

A more detailed description of the SLING parser can be found in this paper:

* Michael Ringgaard, Rahul Gupta, and Fernando C. N. Pereira. 2017.
  *SLING: A framework for frame semantic parsing*. http://arxiv.org/abs/1710.07032.

This is SEMPAR, the first generation of the SLING parser. We have started to work on CASPAR, the second generation of the parser, which can be found [here](https://github.com/google/sling/tree/caspar).

## Trying out the parser

If you just want to try out the parser on a pre-trained model, you can install
the wheel with pip and download a pre-trained parser model. On a Linux machine
with Python 2.7 you can install a pre-built wheel:

```
sudo pip install http://www.jbox.dk/sling/sling-1.0.0-cp27-none-linux_x86_64.whl
```
and download the pre-trained model:
```
wget http://www.jbox.dk/sling/sempar.flow
```
You can then use the parser in Python:
```
import sling

parser = sling.Parser("sempar.flow")

text = raw_input("text: ")
doc = parser.parse(text)
print doc.frame.data(pretty=True)
for m in doc.mentions:
  print "mention", doc.phrase(m.begin, m.end)
```

## Installation

First, make sure that the repository is cloned with `--recursive`, so that you
get all the submodules.

```shell
git clone --recursive https://github.com/google/sling.git
```

The parser trainer uses Tensorflow for training. SLING uses the Python 2.7
distribution of Tensorflow, so this needs to be installed. The installed version
of protocol buffers needs to match the version used by Tensorflow. Finally,
SLING uses [Bazel](https://bazel.build/) as the build system, so you need to
install Bazel in order to build the SLING parser.

```shell
sudo pip install -U protobuf==3.4.0
sudo pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp27-none-linux_x86_64.whl
```

## Building

Operating system: Linux<br>
Languages: C++ 11 (GCC 4), Python 2.7, assembler<br>
CPU: Intel x64 or compatible<br>
Build system: Bazel<br>

You can test your installation by building a few important targets.

```shell
bazel build -c opt sling/nlp/parser sling/nlp/parser/tools:all
```

**NOTE:** In case you get compile errors complaining about missing Tensorflow
includes, try the following:
*  Recreate [this soft
   link](sling/blob/master/third_party/tensorflow/include) to point to your Tensorflow include folder.
*  Change [this
   dependency](https://github.com/google/sling/blob/04d6f28269bdc7d29c71d8dc24d74fe39641f589/third_party/tensorflow/BUILD#L21) to point to your Tensorflow's pywrap library.

## Training

Training a new model consists of preparing the commons store and the training
data, specifying various options and hyperparameters in the training script,
and tracking results as training progresses. These are described below in
detail.

### Data preparation

The first step consists of preparing the commons store (also called global store). This has frame and
schema definitions for all types and roles of interest, e.g.
`/saft/person` or `/pb/love-01` or `/pb/arg0`. In order to build the commons store
for the OntoNotes-based parser you need to checkout PropBank in a directory
parallel to the SLING directory:

```shell
cd ..
git clone https://github.com/propbank/propbank-frames.git propbank
cd sling
sling/nlp/parser/tools/build-commons.sh
```

This will build a SLING store with all the schemas needed and put it into
`/tmp/commons`.

Next, write a converter to convert documents in your existing format to
[SLING documents](sling/nlp/document/document.h). A SLING document is just a
document frame of type `/s/document`. An example of such a frame in textual encoding
can be seen below. It is best to create one SLING document per input sentence.

```shell
{
  :/s/document
  /s/document/text: "John loves Mary"
  /s/document/tokens: [
  {
      :/s/document/token
      /s/token/index: 0
      /s/token/start: 0
      /s/token/length: 4
      /s/token/break: 0
      /s/token/text: "John"
  },
  {
      :/s/document/token
      /s/token/index: 1
      /s/token/start: 5
      /s/token/length: 5
      /s/token/text: "loves"
  },
  {
      :/s/document/token
      /s/token/index: 2
      /s/token/start: 11
      /s/token/length: 4
      /s/token/text: "Mary"
  }]
  /s/document/mention: {=#1
    :/s/phrase
    /s/phrase/begin: 0
    /s/phrase/evokes: {=#2 :/saft/person }
  }
  /s/document/mention: {=#3
    :/s/phrase
    /s/phrase/begin: 1
    /s/phrase/evokes: {=#4
      :/pb/love-01
      /pb/arg0: #2
      /pb/arg1: {=#5 :/saft/person }
    }
  }
  /s/document/mention: {=#6
    :/s/phrase
    /s/phrase/begin: 2
    /s/phrase/evokes: #5
  }
}
```
For writing your converter or getting a better hold of the concepts of frames and store in SLING, you can have a look at detailed deep dive on frames and stores [here](sling/frame/README.md).

The SLING [Document class](sling/nlp/document/document.h)
also has methods to incrementally make such document frames, e.g.
```c++
Store global;
// Read global store from a file via LoadStore().

// Lookup handles in advance.
Handle h_person = global.Lookup("/saft/person");
Handle h_love01 = global.Lookup("/pb/love-01");
Handle h_arg0 = global.Lookup("/pb/arg0");
Handle h_arg1 = global.Lookup("/pb/arg1");

// Prepare the document.
Store store(&global);
Document doc(&store);  // empty document

// Add token information.
doc.SetText("John loves Mary");
doc.AddToken(0, 4, "John", 0);
doc.AddToken(5, 10, "loves", 1);
doc.AddToken(11, 15, "Mary", 1);

// Create frames that will eventually be evoked.
Builder b1(&store);
b1.AddIsA(h_person);
Frame john_frame = b1.Create();

Builder b2(&store);
b2.AddIsA(h_person);
Frame mary_frame = b2.Create();

Builder b3(&store);
b3.AddIsA(h_love01);
b3.Add(h_arg0, john_frame);
b3.Add(h_arg1, mary_frame);
Frame love_frame = b3.Create();

# Add spans and evoke frames from them.
doc.AddSpan(0, 1)->Evoke(john_frame);
doc.AddSpan(1, 2)->Evoke(love_frame);
doc.AddSpan(2, 3)->Evoke(mary_frame);

doc.Update();
string encoded = Encode(doc.top());

// Write 'encoded' to a zip stream or a file.
```

Use the converter to create the following corpora:
+ Training corpus of annotated SLING documents.
+ Dev corpus of annotated SLING documents.

  The default corpus format is zip, where the zip file contains one file
  per document, and the file for a document is just its encoded document
  frame. An alternate format is to have a folder with one file per
  document. More formats can be added by modifying the reader code
[here](https://github.com/google/sling/blob/88771ebb771d2e32a2f481d3523c4747303047e0/nlp/document/document-source.cc#L107) and [here](https://github.com/google/sling/blob/0c8ec1dcc4057c64eac8f8d5939b128a10750c63/nlp/parser/tools/train.py#L57).

### Specify training options and hyperparameters:

Once the commons store and the corpora have been built, you are ready for training
a model. For this, use the supplied [training script](sling/nlp/parser/tools/train.sh).
The script provides various commandline arguments. The ones that specify
the input data are:
+ `--commons`: File path of the commons store built in the previous step.
+ `--train`: Path to the training corpus built in the previous step.
+ `--dev`: Path to the annotated dev corpus built in the previous step.
+ `--output` or `--output_dir`: Output folder where checkpoints, master spec,
  temporary files, and the final model will be saved.

Then we have the various training options and hyperparameters:
+ `--oov_features`: Whether fallback lexical features should be used in the LSTMs.
+ `--word_embeddings`: Empty, or path to pretrained word embeddings in
  [Mikolov's word2vec format](https://github.com/tmikolov/word2vec/blob/master/word2vec.c).
  If supplied, these are used to initialize the embeddings for word features.
+ `--word_embeddings_dim`: Dimensionality of embeddings for word features.
  Should be the same as the pretrained embeddings, if they are supplied.
+ `--batch`: Batch size used during training.
+ `--report_every`: Checkpoint interval.
+ `--train_steps`: Number of training steps.
+ `--method`: Optimization method to use (e.g. adam or momentum), along
  with auxiliary arguments like `--adam_beta1`, `--adam_beta2`, `--adam_eps`.
+ `--dropout_keep_rate`: Probability of keeping after dropout during
  training , so `--dropout_keep_rate=1.0` means nothing is dropped.
+ `--learning_rate`: Learning rate.
+ `--decay`: Decay steps.
+ `--grad_clip_norm`: Max norm beyond which gradients will be clipped.
+ `--moving_average`: Whether or not to use exponential moving average.
+ `--seed`, `--seed2`: Randomization seeds used for initializing embedding matrices.

The script comes with reasonable defaults for the hyperparameters for
training a semantic parser model, but it would be a good idea to hardcode
your favorite arguments [directly in the
script](https://github.com/google/sling/blob/0c8ec1dcc4057c64eac8f8d5939b128a10750c63/nlp/parser/tools/train.sh#L51)
to avoid supplying them again and again on the commandline.

### Run the training script

To test your training setup, you can kick off a small training run:
```shell
./sling/nlp/parser/tools/train.sh --report_every=500 --train_steps=1000
```

This training run should be over in 10-20 minutes, and should checkpoint and
evaluate after every 500 steps. For a full-training run, we suggest increasing
the number of steps to something like 100,000 and decreasing the checkpoint
frequency to something like every 2000-5000 steps.

As training proceeds, the training script produces a lot of useful
diagnostic information, which we describe below.

* The script begins by constructing an action table, which is a list of all
  transitions required to generate the gold frames in the training corpus.
  The table and its summary are dumped in `$OUTPUT_FOLDER/{table,
  table.summary}`, and this path is logged by the script in its output.
  For example, here is the action table summary for the
  semantic parsing model included in this release:

  ```shell
  $ cat $OUTPUT_FOLDER/table.summary

  Actions Summary
  ===================================================
  Action Type || Unique Arg Combinations || Raw Count
  ===================================================
      OVERALL ||                   6,968 || 4,038,809
         STOP ||                       1 ||   111,006
        SHIFT ||                       1 || 2,206,274
       ASSIGN ||                      13 ||     5,430
      CONNECT ||                   1,421 ||   635,734
        EVOKE ||                   5,532 || 1,080,365
  ===================================================
  <snip>
  ```

* The script then prepares all the lexical resources (e.g. word
  vocabulary, affixes etc), and the DRAGNN MasterSpec protocol buffer, which
  completely specifies the configuration of the two LSTMs and the feed forward
  unit, including the features used by each component, and the dimensions of
  the various embeddings.

  If you wish to modify the default set of features,
  then you would have to modify the [MasterSpec generation code](https://github.com/google/sling/blob/88771ebb771d2e32a2f481d3523c4747303047e0/nlp/parser/trainer/generate-master-spec.cc#L319)
  and add any new feature definitions [here](https://github.com/google/sling/blob/88771ebb771d2e32a2f481d3523c4747303047e0/nlp/parser/trainer/feature-extractor.cc#L73)
  and/or [here](https://github.com/google/sling/blob/88771ebb771d2e32a2f481d3523c4747303047e0/nlp/parser/trainer/feature-extractor.cc#L193). Recall however that
  `--word_embeddings_dim`, `--pretrained_embeddings`, and `--oov_features`
  allow you to do some of this directly from the commandline.

  Once the MasterSpec is ready, the script would log that the spec is
  being dumped at `$OUTPUT_FOLDER/master_spec` as a textualized protocol
  buffer, so you can visually check whether the configuration looks good or not.

  **NOTE:** If you specify `--spec_only` on the commandline, then the script
  will finish here. This is useful for first ascertaining that the spec and
  action table look right, particularly while debugging or running a new
  training setup for the first time.

* The script will now generate the Tensorflow graph, and log some messages
  about the internal structure of the graph. Once that is done, it will inform
  that it's creating a log directory for running [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

  ```shell
  Wrote events (incl. graph) for Tensorboard to folder: /my/output/folder/tensorboard
  The graph can be viewed via
  tensorboard --logdir=/my/output/folder/tensorboard
  then navigating to http://localhost:6006 and clicking on 'GRAPHS'
  ```

  Tensorboard is a useful tool that provides a browser-based UI to track
  training, particularly the evaluation metrics at various checkpoints.
  It also allows you to view the Tensorflow graph used for training.

* Once the graph is ready, training will commence, and you will see
  the training cost being logged at regular intervals.
  ```shell
  INFO:tensorflow:Initial cost at step 0: 2.949682
  INFO:tensorflow:cost at step 100: 1.218417
  INFO:tensorflow:cost at step 200: 0.991216
  INFO:tensorflow:cost at step 300: 0.838045
  <snip>
  ```
  After every checkpoint interval (specified via `--report_every`),
  it will save the model and evaluate it on the dev corpus.
  The evaluation runs a [graph matching algorithm](sling/nlp/parser/trainer/frame-evaluation.h)
  that outputs various metrics from aligning the gold frame graph
  vs the test frame graph. If you are looking for a single number to
  quantify your model, then we suggest using **SLOT_F1**, which aggregates across
  frame type and role accuracies (i.e. both node and edge alignment scores).

  Note that graph matching is an intrinsic evaluation, so if you wish to swap
  it with an extrinsic evaluation, then just replace the binary
  [here](https://github.com/google/sling/blob/0c8ec1dcc4057c64eac8f8d5939b128a10750c63/nlp/parser/tools/train.py#L97) with your evaluation binary.

* Finally, the best performing checkpoint will be converted into a Myelin flow file
  in `$OUTPUT_FOLDER/sempar.flow`.

  **NOTE:** A common use-case is that one often wants to play around with
  different training options without really wanting to change the spec,
  or the action table, or any lexical resources. For this, use the
  `--train_only` commandline argument. This will initiate training from
  the Tensorflow graph generation step, and will use the pre-generated spec
  etc. from the same output folder.

We have made a synthetic training and evaluation corpus available for trying out the parser
trainer:

```
curl -o /tmp/conll-2003-sempar.tar.gz http://www.jbox.dk/sling/conll-2003-sempar.tar.gz
tar -xvf /tmp/conll-2003-sempar.tar.gz
```

See [local/conll2003/README.md](local/conll2003/README.md) for instructions on how to train a parser.

## Parsing

The trained parser model is stored in a [Myelin](sling/myelin/README.md) flow file,
e.g. `sempar.flow`. It contains all the information needed for parsing text:
* The neural network units (LR, RL, FF) with the parameters learned from
training.
* Feature maps for the lexicon and affixes.
* The commons store is a [SLING store](sling/frame/README.md) with the schemas for the
frames.
* The action table with all the transition actions.

A pre-trained model can be download from [here](http://www.jbox.dk/sling/sempar.flow).
The model can be loaded and initialized in the following way:

```c++
#include "sling/frame/store.h"
#include "sling/nlp/document/document-tokenizer.h"
#include "sling/nlp/parser/parser.h"

// Load parser model.
sling::Store commons;
sling::nlp::Parser parser;
parser.Load(&commons, "/tmp/sempar.flow");
commons.Freeze();

// Create document tokenizer.
sling::nlp::DocumentTokenizer tokenizer;
```

In order to parse some text, it first needs to be tokenized. The document with
text, tokens, and frames is stored in a local document frame store.

```c++
// Create frame store for document.
sling::Store store(&commons);
sling::nlp::Document document(&store);

// Tokenize text.
string text = "John hit the ball with a bat.";
tokenizer.Tokenize(&document, text);

// Parse document.
parser.Parse(&document);
document.Update();

// Output document annotations.
std::cout << sling::ToText(document.top(), 2);
```

## Annotation Tools

SLING comes with utility tools for annotating a corpus of documents with frames
using a parser model, benchmarking this annotation process, and optionally
evaluating the annotated frames against supplied gold frames.

We provide two such tools -- a
[tf-parse](sling/nlp/parser/tools/tf-parse.py) Python script, and a [Myelin-based
parser tool](sling/nlp/parser/tools/parse.cc).
Given the same trained parser model, both these tools should produce
the same annotated frames and evaluation numbers. However the Myelin-based
parser is significantly faster than Tensorflow-based tf-parse ([3x-10x in our
experiments](http://www.jbox.dk/sling/sempar-profile.htm)).

### Myelin-based parser tool

This tool takes the following commandline arguments:

*  `--parser` : This should point to a Myelin flow, e.g. one created by the
   training script.
*  If `--text` is specified then the parser is run over the supplied text, and
   prints the annotated frame(s) in text mode. The indentation of the text
   output can be controlled by `--indent`. E.g.
   ```shell
   bazel build -c opt sling/nlp/parser/tools:parse
   bazel-bin/sling/nlp/parser/tools/parse --logtostderr \
      --parser=<path to flow file> --text="John loves Mary" --indent=2

   {=#1
     :/s/document
     /s/document/text: "John loves Mary"
     /s/document/tokens: [{=#2
       :/s/token
       /s/token/index: 0
       /s/token/text: "John"
       /s/token/start: 0
       /s/token/length: 4
       /s/token/break: 0
     }, {=#3
       :/s/token
       /s/token/index: 1
       /s/token/text: "loves"
       /s/token/start: 5
       /s/token/length: 5
     }, {=#4
       :/s/token
       /s/token/index: 2
       /s/token/text: "Mary"
       /s/token/start: 11
       /s/token/length: 4
     }]
     /s/document/mention: {=#5
       :/s/phrase
       /s/phrase/begin: 0
       /s/phrase/evokes: {=#6
         :/saft/person
       }
     }
     /s/document/mention: {=#7
       :/s/phrase
       /s/phrase/begin: 1
       /s/phrase/evokes: {=#8
         :/pb/love-01
         /pb/arg0: #6
         /pb/arg1: {=#9
           :/saft/person
         }
       }
     }
     /s/document/mention: {=#10
       :/s/phrase
       /s/phrase/begin: 2
       /s/phrase/evokes: #9
     }
   }
   I0927 14:44:25.705880 30901 parse.cc:154] 823.732 tokens/sec
   ```
*  If `--benchmark` is specified then the parser is run on the document
   corpus specified via `--corpus`. This corpus should be prepared similarly to
   how the training/dev corpora were created. The processing can be limited to
   the first N documents by specifying `--maxdocs=N`.

   ```shell
    bazel-bin/sling/nlp/parser/tools/parse --logtostderr \
      --parser=sempar.flow --corpus=dev.zip -benchmark --maxdocs=200

    I0927 14:45:36.634670 30934 parse.cc:127] Load parser from sempar.flow
    I0927 14:45:37.307870 30934 parse.cc:135] 565.077 ms loading parser
    I0927 14:45:37.307922 30934 parse.cc:161] Benchmarking parser on dev.zip
    I0927 14:45:39.059257 30934 parse.cc:184] 200 documents, 3369 tokens, 2289.91 tokens/sec
   ```

   If `--profile` is specified, the parser will run with profiling
   instrumentation enabled and output a detailed profile report with execution
   timing for each operation in the neural network.

*  If `--evaluate` is specified then the tool expects `--corpora` to specify
   a corpora with gold frames. It then runs the parser model over a frame-less
   version of this corpora and evaluates the annotated frames vs the gold
   frames. Again, one can use `--maxdocs` to limit the evaluation to the first N
   documents.
   ```shell
   bazel-bin/sling/nlp/parser/tools/parse --logtostderr \
     --evaluate --parser=sempar.flow --corpus=dev.zip --maxdocs=200

   I0927 14:51:39.542151 31336 parse.cc:127] Load parser from sempar.flow
   I0927 14:51:40.211920 31336 parse.cc:135] 562.249 ms loading parser
   I0927 14:51:40.211973 31336 parse.cc:194] Evaluating parser on dev.zip
   SPAN_P+ 1442
   SPAN_P- 93
   SPAN_R+ 1442
   SPAN_R- 133
   SPAN_Precision  93.941368078175884
   SPAN_Recall     91.555555555555557
   SPAN_F1 92.733118971061089
   ...
   <snip>
   ...
   SLOT_F1 78.398993883366586
   COMBINED_P+     4920
   COMBINED_P-     633
   COMBINED_R+     4923
   COMBINED_R-     901
   COMBINED_Precision      88.60075634792004
   COMBINED_Recall 84.529532967032978
   COMBINED_F1     86.517276488704127
   ```

### Tensorflow-based parser tool

An alternative to running the Myelin-based parsing tool is to run the tf-parse
Python script that executes the annotation part of the Tensorflow graph over the
input documents. It takes the following arguments:

*  `--parser_dir`: This should be the directory where the trained model is saved.
*  `--commons`: Path to the commons store. Should be the same as the one used in training.
*  `--corpus`: Corpus of documents that will be annotated with the model.
*  `--batch`: Batch size. Higher batch sizes are efficient but only if all the batch
    documents are roughly of similar length.
*  `--threads` : Number of threads to use in Tensorflow. This drives Tensorflow's
    inter-op and intra-op parallelism. Making this very high will lead to inefficiencies
    due to inter-thread CPU contention.
*  `--output`: (Optional) File name where the annotated corpus will be saved.
*  `--evaluate`: (Optional) If true, then it will evaluate the annotated corpus vs
    the gold corpus (specified via --corpus).

Sample Usage:
```shell
python sling/nlp/parser/tools/tf-parse.py \
  --parser_dir=/path/to/training/script/output/folder \
  --commons=/path/to/commons \
  --corpus=/path/to/gold/eval/corpus \
  --batch=512 \
  --threads=4 \
  --output=annotated.zip \
  --evaluate
```

## Credits

Original authors of the code in this package include:

*   Michael Ringgaard
*   Rahul Gupta



