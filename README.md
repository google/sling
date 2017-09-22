# SLING - A natural language frame semantics parser

SLING is a parser for annotating text with frame semantic annotations. It is
trained on an annotated corpus using [Tensorflow](https://www.tensorflow.org/)
and [Dragnn](https://github.com/tensorflow/models/blob/master/syntaxnet/g3doc/DRAGNN.md).

## Installation

The parser trainer uses Tensorflow for training. SLING uses the Python 2.7
distribution of Tensorflow, so this needs to be installed. The installed version
of protocol buffers needs to match the version used by Tensorflow.

```shell
sudo pip install -U protobuf==3.3.0
sudo pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp27-none-linux_x86_64.whl
```

## Building

Operating system: Linux<br>
Languages: C++, Python 2.7, assembler<br>
CPU: Intel x64 or compatible<br>
Build system: Bazel<br>

SLING uses [Bazel](https://bazel.build/) as build system, so you need to install
Bazel in order to build the SLING parser.

```shell
bazel build -c opt nlp/parser
```

## Training

Training a new model consists of preparing the commons store and the training
data, specifying various options and hyperparameters in the training script,
and tracking results as training progresses. These are described below in
detail.

### Data preparation

The first step consists of preparing the commons store (also called global store). This has frame and
schema definitions for all types and roles of interest, e.g.
`/s/person` or `/pb/love-01` or `/pb/arg0`. More details on store
creation can be found [here (TBD)](TBD).

Next, write a converter to convert documents in your existing format to
[SLING documents](https://github.com/google/sling/blob/master/nlp/document/document.h). A SLING document is just a
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
    /s/phrase/evokes: {=#2 :/s/person }
  }
  /s/document/mention: {=#3
    :/s/phrase
    /s/phrase/begin: 1
    /s/phrase/evokes: {=#4
      :/pb/love-01
      /pb/arg0: #2
      /pb/arg1: {=#5 :/s/person }
    }
  }
  /s/document/mention: {=#6
    :/s/phrase
    /s/phrase/begin: 2
    /s/phrase/evokes: #5
  }
}
```

The SLING [Document class](https://github.com/google/sling/blob/master/nlp/document/document.h)
also has methods to incrementally make such document frames, e.g.
```c++
Store global;
// Read global store from a file via LoadStore().

// Lookup handles in advance.
Handle h_person = global.Lookup("/s/person");
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
+ A version of the dev corpus without any annotations, i.e. a SLING
  document in this corpus will only have token and text information.
  The documents in this corpus should be in the same order as in the
  annotated dev corpus.

  The default corpus format is zip, where the zip file contains one file
  per document, and the file for a document is just its encoded document
  frame. An alternate format is to have a folder with one file per
  document. More formats can be added by modifying the reader code
[here](https://github.com/google/sling/blob/88771ebb771d2e32a2f481d3523c4747303047e0/nlp/document/document-source.cc#L107) and [here](https://github.com/google/sling/blob/88771ebb771d2e32a2f481d3523c4747303047e0/nlp/parser/trainer/train.py#L57).

### Specify training options and hyperparameters:

Once the commons store and the corpora have been built, you are ready for training
a model. For this, use the supplied [training script](https://github.com/google/sling/blob/master/nlp/parser/trainer/train.sh).
The script provides various commandline arguments. The ones that specify
the input data are:
+ `--commons`: File path of the commons store built in the previous step.
+ `--train`: Path to the training corpus built in the previous step.
+ `--dev`: Path to the annotated dev corpus built in the previous step.
+ `--dev_without_gold`: Path to the dev corpus without annotations built
  in the previous step.
+ `--output` or `--output_dir`: Output folder where checkpoints, master spec,
  temporary files, and the final model will be saved.

Then we have the various training options and hyperparameters:
+ `--oov_features`: Whether fallback lexical features should be used in the LSTMs.
+ `--word_embeddings`: Empty or path to pretrained word embeddings in
  Tensorflow's RecordIO format. If supplied, these are used to initialize
  the embeddings for word features.
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
training a semantic parser model, but it would be a good idea to hard
code your favorite arguments [directly in the
script](https://github.com/google/sling/blob/88771ebb771d2e32a2f481d3523c4747303047e0/nlp/parser/trainer/train.sh#L53)
to avoid supplying them again and again on the commandline.

### Run the training script
```shell
./nlp/parser/trainer/train.sh --report_every=1000 --train_steps=100000
```

As training proceeds, it produces a lot of useful
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
  The evaluation runs a [graph matching algorithm](https://github.com/google/sling/blob/master/nlp/parser/trainer/frame-evaluation.h)
  that outputs various metrics from aligning the gold frame graph
  vs the test frame graph. If you are looking for a single number to
  quantify your model, then we suggest using **SLOT_F1**, which aggregates across
  frame type and role accuracies (i.e. both node and edge alignment scores).

  Note that graph matching is an intrinsic evaluation, so if you wish to swap
  it with an extrinsic evaluation, then just replace the binary
  [here](https://github.com/google/sling/blob/88771ebb771d2e32a2f481d3523c4747303047e0/nlp/parser/trainer/train.py#L97) with your evaluation binary.

* Finally, if `--flow` is specified, then the best performing checkpoint
  will be converted into a Myelin flow file (more on Myelin and its flows
  in the next section).
  This will enable its use in a fast Myelin-based parser runtime.

  **NOTE:** A common use-case is that one often wants to play around with
  different training options without really wanting to change the spec,
  or the action table, or any lexical resources. For this, use the
  `--train_only` commandline argument. This will initiate training from
  the Tensorflow graph generation step, and will use the pre-generated spec
  etc. from the same output folder.

## Parsing

The parser needs a frame store and a parser model. The [frame store](frame/README.md)
contains the the schemas for the frames and an action table produced by the
parser trainer. The parser model contains the trained neural network for the
model. The parser model is stored in a [Myelin](myelin/README.md) flow file.

The model can be loaded and initialized in the following way:

```c++
#include "frame/store.h"
#include "frame/serialization.h"
#include "nlp/document/document-tokenizer.h"
#include "nlp/parser/parser.h"

// Initialize global frame store.
sling::Store commons;
sling::FileDecoder decoder(&commons, "/tmp/parser.sling");
decoder.DecodeAll();

// Load parser model.
sling::nlp::Parser parser;
parser.Load(&commons, "/tmp/parser.flow");
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

## Credits

Original authors of the code in this package include:

*   Michael Ringgaard
*   Rahul Gupta



