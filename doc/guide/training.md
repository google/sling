## Training a SLING Parser

Training a new model consists of preparing the commons store and the training
data, specifying various options and hyperparameters in the training script,
and tracking results as training progresses. These are described below in
detail.

### Data preparation

SLING's training data consists of training and dev documents formatted as SLING
frames. These document frames in turn consist of frames that have the desired types (e.g.
/saft/person, /saft/location, /pb/love-01) and contain the desired roles
(e.g. /pb/arg0, /pb/argm-loc). These frame types and roles, and any schema that
organizes them is stored separately in a 'commons store'. Therefore the complete
training data consists of (a) a commons store, (b) a recordio of training
documents, (c) a recordio of dev documents.

SLING offers users tools to either build the commons store themselves, or
automatically create it from the training/dev documents. Building the commons
store yourself allows you to store rich hierarchy and schema information
about types and roles in the commons store.
In case this schema is not needed/exploited, one can instead use the automatic
commons construction tool that collects all types and roles
from the documents and puts them in a commons store. Below we describe
both these tools.

#### Option 1: Automatically building the commons store from documents.
Here, we just create a corpus of documents that link to frames using types and
roles that will be automatically collected later. A typical SLING training
document is just a frame of type `document`. It is best to create one
SLING document per input sentence. The textual encoding of a sample SLING
document is shown below:

```shell
{
  :document
  text: "John loves Mary"
  tokens: [
    {start: 0 size: 4},
    {start: 5 size: 5},
    {start: 11 size: 4}
  ]
  mention: {=#1
    begin: 0
    evokes: {=#2 :/saft/person }
  }
  mention: {=#3
    begin: 1
    evokes: {=#4
      :/pb/love-01
      /pb/arg0: #2
      /pb/arg1: {=#5 :/saft/person }
    }
  }
  mention: {=#6
    begin: 2
    evokes: #5
  }
}
```

SLING documents like these can be built programmatically in C++ using
the [Document class](../../sling/nlp/document/document.h), or via the
[SLING Python API](pyapi.md). In either case, all document frames are written
to a single file in SLING's [recordio file
format](../../sling/file/recordio.h), where a single record corresponds to one
encoded document. This file format is up to 25x faster to read than zip files,
yet provides almost identical compression ratios.

We now illustrate SLING document preparation using the C++ API.
For writing your converter or getting a better hold of the concepts of frames
and store in SLING, you can have a look at detailed deep dive on frames and
stores [here](frames.md).

```c++
Store commons;

// If we have a commons store already built, then do:
// LoadStore(<filename>, &commons);

commons.Freeze();

// Writer that will write all documents to a recordio file.
RecordWriter writer(<filename>);

// BEGIN: Document preparation. This needs to be done per document.
Store store(&commons); // document will be created in its own local store

// Lookup the types and roles that will be used in this document.
// These will be first looked up in 'store', but since 'store' is empty,
// they will then be looked up in 'commons'. If 'commons' was already
// constructed and had these symbols, their handles would be returned.
// If not, these symbols will be added as unbound symbols to 'store'.
Handle h_person = store.Lookup("/saft/person");
Handle h_love01 = store.Lookup("/pb/love-01");
Handle h_arg0 = store.Lookup("/pb/arg0");
Handle h_arg1 = store.Lookup("/pb/arg1");

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

// Finish the document.
doc.Update();

// Write binary encoding of the document to the recordio writer.
string encoded = Encode(doc.top());
writer.Write(encoded);
// END: Document preparation.

// Close the recordio writer.
writer.Close();
```

The SLING Python API version of document preparation is as follows:

```python
import sling

commons = sling.Store()

# If we have a commons already built, then do the following:
# commons.load(<filename>)

schema = sling.DocumentSchema(commons)
commons.freeze()

# Initialize recordio writer.
writer = sling.RecordWriter("/tmp/train.rec")

# BEGIN: Document preparation. Needs to done for every document.
store = sling.Store(commons)
doc = sling.Document(None, store, schema)

# Add tokens.
doc.add_token("John")
doc.add_token("loves")
doc.add_token("Mary")

# Lookup/create types and roles used in the document.
person = store["/saft/person"]
love = store["/pb/love-01"]
arg0 = store["/pb/arg0"]
arg1 = store["/pb/arg1"]

# Evoke frames from mentions.
john_frame = doc.evoke_type(0, 1, person)
mary_frame = doc.evoke_type(2, 3, person)
love_frame = doc.evoke_type(1, 2), love)

# Add roles to frames.
love_frame[arg0] = john_frame
love_frame[arg1] = mary_frame

# Write the document to the recordio writer.
doc.update()
writer.write("unused_key", doc.frame.data(binary=True))
# END: Document preparation.

# Close recordio writer.
writer.close()
```

##### (Optional): Commons store construction from documents.

Once the training and dev recordios are created, one can use
[this script](../../sling/nlp/parser/tools/commons_from_corpora.py) to
create a commons store from them. Note that explicit creation of this commons
store is NOT needed for training, since the training script will anyway invoke
the same script to create the commons store behind the scenes. But we mention
this here in case one wishes to inspect the automatically created commons.

```shell
python3 sling/nlp/parser/tools/commons_from_corpora.py \
  --input=<path to train.rec>,<path to dev.rec>,<any other rec files> \
  --output=<path where commons will be written>
```

#### Option 2: Manually building the commons store.
This alternative is useful if one wishes to manually construct their own
commons store, containing rich schema information that they plan to exploit.
We illustrate this by building the commons store that was used for Propbank SRL
and entity tagging. In order to build this commons store
you first need to checkout PropBank in a directory
parallel to the SLING directory:

```shell
cd ..
git clone https://github.com/propbank/propbank-frames.git propbank
cd sling
sling/nlp/parser/tools/build-commons.sh
```

This will build a SLING store with all the schemas needed and put it into
`/tmp/commons`.

Another toy example, where we have access to all frame types and roles in
advance, is as follows:

```python
import sling

commons = sling.Store()
schema = sling.DocumentSchema(commons)

frame_types = ["/saft/person", "/saft/location", "/saft/org", "/saft/other"]
frame_roles = ["/role/arg0", "/role/arg1", "/role/argloc", "/role/argtmp"]
for id in frame_types + frame_roles:
  _ = commons.frame(id)  # create a frame in commons with that id

commons.freeze()
commons.save(<filename>, binary=True)
```

Once the commons store is written, it can be loaded via LoadStore() in C++
or load() in Python, and the same code snippets as in Option 1 can be used to
create training and dev documents.

##### (Optional but recommended): Document validation.

In both options, it is highly recommended to validate the prepared documents
for common sources of errors, e.g. crossing spans, nil roles/values, unknown
frame types etc.

```shell
python sling/nlp/parser/tools/validate.py \
  --input=<path to train or dev recordio file> \
  --commons=<path to manually or automatically built commons store>
```

The full list of supported error types is
[here](../../sling/nlp/parser/tools/validate.py#L37).

### Specify training options and hyperparameters:

Once the corpora (and optionally the commons store) have been built, you are ready for training
a model. For this, use the supplied [training script](../../sling/nlp/parser/tools/train.sh).
The script provides various commandline arguments. The ones that specify
the input data are:
+ `--commons`: Optional. File path of the commons store built in the previous
step. If not specified or non-existent, then a commons store will be
automatically built by the training script using Option 1 above.
+ `--train`: Path to the training corpus built in the previous step.
+ `--dev`: Path to the annotated dev corpus built in the previous step.
+ `--output` or `--output_dir`: Output folder where checkpoints,
  temporary files, and the final model will be saved.

Then we have the various training options and hyperparameters:
+ `--word_embeddings`: Empty, or path to pretrained word embeddings in
  [Mikolov's word2vec format](https://github.com/tmikolov/word2vec/blob/master/word2vec.c).
  If supplied, these are used to initialize the embeddings for word features.
+ `--batch`: Batch size used during training.
+ `--report_every`: Checkpoint interval (in number of batches).
+ `--steps`: Number of training batches to process.
+ `--method`: Optimization method to use (e.g. adam or momentum), along
  with auxiliary arguments like `--adam_beta1`, `--adam_beta2`, `--adam_eps`.
+ `--learning_rate`: Learning rate.
+ `--grad_clip_norm`: Max norm beyond which gradients will be clipped.
+ `--moving_average`: Whether or not to use exponential moving average.

The script comes with reasonable defaults for the hyperparameters for
training a semantic parser model, but it would be a good idea to hardcode
your favorite arguments [directly in the
flag definitions](../../sling/nlp/parser/trainer/train_util.py#L94)
to avoid supplying them again and again on the commandline.

### Run the training script

The parser trainer uses PyTorch for training, so it needs to be installed:
```shell
sudo pip3 install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
```

To test your training setup, you can kick off a small training run:
```shell
./sling/nlp/parser/tools/train.sh [--commons=<path to commons>] \
   --train=<oath to train recordio> --dev=<path to dev recordio> \
   --report_every=500 --train_steps=1000 --output=<output folder>
```

This training run should be over in 10-20 minutes, and should checkpoint and
evaluate after every 500 steps. For a full-training run, we suggest increasing
the number of steps to something like 100,000 and decreasing the checkpoint
frequency to something like every 2000-5000 steps.

As training proceeds, the training script produces a lot of useful
diagnostic information, which is logged by default to a file called "log"
inside the specified output folder.

* The script will first generate the PyTorch model and print its specification, i.e. various
sub-modules inside the model and their dimensionalities.

```shell
Modules: Caspar(
  (lr_lstm_embedding_words): EmbeddingBag(53257, 32, mode=sum)
  (rl_lstm_embedding_words): EmbeddingBag(53257, 32, mode=sum)
  (lr_lstm_embedding_suffix): EmbeddingBag(8334, 16, mode=sum)
  (rl_lstm_embedding_suffix): EmbeddingBag(8334, 16, mode=sum)
  (lr_lstm_embedding_capitalization): EmbeddingBag(5, 8, mode=sum)
  (rl_lstm_embedding_capitalization): EmbeddingBag(5, 8, mode=sum)
  (lr_lstm_embedding_hyphen): EmbeddingBag(2, 8, mode=sum)
  (rl_lstm_embedding_hyphen): EmbeddingBag(2, 8, mode=sum)
  (lr_lstm_embedding_punctuation): EmbeddingBag(3, 8, mode=sum)
  (rl_lstm_embedding_punctuation): EmbeddingBag(3, 8, mode=sum)
  (lr_lstm_embedding_quote): EmbeddingBag(4, 8, mode=sum)
  (rl_lstm_embedding_quote): EmbeddingBag(4, 8, mode=sum)
  (lr_lstm_embedding_digit): EmbeddingBag(3, 8, mode=sum)
  (rl_lstm_embedding_digit): EmbeddingBag(3, 8, mode=sum)
  (lr_lstm): DragnnLSTM(in=88, hidden=256)
  (rl_lstm): DragnnLSTM(in=88, hidden=256)
  (ff_fixed_embedding_in-roles): EmbeddingBag(125, 16, mode=sum)
  (ff_fixed_embedding_out-roles): EmbeddingBag(125, 16, mode=sum)
  (ff_fixed_embedding_labeled-roles): EmbeddingBag(625, 16, mode=sum)
  (ff_fixed_embedding_unlabeled-roles): EmbeddingBag(25, 16, mode=sum)
  (ff_link_transform_frame-creation-steps): LinkTransform(input_activation=128, dim=64, oov_vector=64)
  (ff_link_transform_frame-focus-steps): LinkTransform(input_activation=128, dim=64, oov_vector=64)
  (ff_link_transform_frame-end-lr): LinkTransform(input_activation=256, dim=32, oov_vector=32)
  (ff_link_transform_frame-end-rl): LinkTransform(input_activation=256, dim=32, oov_vector=32)
  (ff_link_transform_history): LinkTransform(input_activation=128, dim=64, oov_vector=64)
  (ff_link_transform_lr): LinkTransform(input_activation=256, dim=32, oov_vector=32)
  (ff_link_transform_rl): LinkTransform(input_activation=256, dim=32, oov_vector=32)
  (ff_layer): Projection(in=1344, out=128, bias=True)
  (ff_relu): ReLU()
  (ff_softmax): Projection(in=128, out=6968, bias=True)
  (loss_fn): CrossEntropyLoss(
  )
)

```

* Training will now commence, and you will see the training cost being logged at regular intervals.
  ```shell
  BatchLoss after (1 batches = 8 examples): 2.94969940186  incl. L2= [0.000000029] (1.6 secs) <snip>
  BatchLoss after (2 batches = 16 examples): 2.94627690315  incl. L2= [0.000000029] (1.9 secs) <snip>
  BatchLoss after (3 batches = 24 examples): 2.94153237343  incl. L2= [0.000000037] (1.1 secs) <snip>
  ...
  <snip>

  ```
* After every checkpoint interval (specified via `--report_every`),
  it will save the model and evaluate it on the dev corpus.
  The evaluation runs a [graph matching algorithm](../../sling/nlp/parser/trainer/frame-evaluation.h)
  that outputs various metrics from aligning the gold frame graph
  vs the test frame graph. If you are looking for a single number to
  quantify your model, then we suggest using **SLOT_F1**, which aggregates across
  frame type and role accuracies (i.e. both node and edge alignment scores).

  Note that graph matching is an intrinsic evaluation, so if you wish to swap
  it with an extrinsic evaluation, then just replace the binary
  [here](../../sling/nlp/parser/trainer/train_util.py#L30) with your evaluation binary.

* At any point, the best performing checkpoint will be available as a Myelin flow file
  in `<output folder>/caspar.best.flow`.

**NOTE:**
* If you wish to modify the default set of features,
then you would have to modify the [feature specification code](../../sling/nlp/parser/trainer/spec.py#L212).

