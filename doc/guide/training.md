## Training a SLING Parser

Training a new model consists of preparing the commons store and the training
data, specifying various options and hyperparameters in the training script,
and tracking results as training progresses. These are described below in
detail.

### Data preparation

The first step consists of preparing the commons store (also called global store).
This has frame and schema definitions for all types and roles of interest, e.g.
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
[SLING documents](../../sling/nlp/document/document.h). A SLING document is 
just a document frame of type `/s/document`. An example of such a frame in 
textual encoding can be seen below. It is best to create one SLING document per 
input sentence.

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
For writing your converter or getting a better hold of the concepts of frames 
and store in SLING, you can have a look at detailed deep dive on frames and 
stores [here](frames.md).

The SLING [Document class](../../sling/nlp/document/document.h)
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

// Append 'encoded' to a recordio file.
RecordWriter writer(<filename>);

writer.Write(encoded);
...<write more documents>

writer.Close();
```

Use the converter to create the following corpora:
+ Training corpus of annotated SLING documents.
+ Dev corpus of annotated SLING documents.

CASPAR uses the [recordio file format](../../sling/file/recordio.h) for training 
where each record corresponds to one encoded document. This format is up to 25x 
faster to read than zip files, with almost identical compression ratios.

### Specify training options and hyperparameters:

Once the commons store and the corpora have been built, you are ready for training
a model. For this, use the supplied [training script](../../sling/nlp/parser/tools/train.sh).
The script provides various commandline arguments. The ones that specify
the input data are:
+ `--commons`: File path of the commons store built in the previous step.
+ `--train`: Path to the training corpus built in the previous step.
+ `--dev`: Path to the annotated dev corpus built in the previous step.
+ `--output` or `--output_dir`: Output folder where checkpoints, master spec,
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

To test your training setup, you can kick off a small training run:
```shell
./sling/nlp/parser/tools/train.sh --commons=<path to commons> \
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
Modules: Sempar(
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
  in `<output folder>/pytorch.best.flow`.

**NOTE:**
* If you wish to modify the default set of features,
then you would have to modify the [feature specification code](../../sling/nlp/parser/trainer/spec.py#L212).

