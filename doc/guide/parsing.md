## Parsing with SLING

The trained parser model is stored in a [Myelin](myelin.md) flow file,
It contains all the information needed for parsing text:
* The neural network units (LR, RL, FF) with the parameters learned from
training.
* Feature maps for the lexicon and affixes.
* The commons store is a [SLING store](frames.md) with the schemas for the
frames.
* The action table with all the transition actions.

A pre-trained model can be downloaded from [here](http://www.jbox.dk/sling/sempar.flow).
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

## Myelin-based parser tool

SLING comes with a [parsing tool](../../sling/nlp/parser/tools/parse.cc)
for annotating a corpus of documents with frames
using a parser model, benchmarking this annotation process, and optionally
evaluating the annotated frames against supplied gold frames.


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
     --evaluate --parser=sempar.flow --corpus=dev.rec --maxdocs=200

   I0927 14:51:39.542151 31336 parse.cc:127] Load parser from sempar.flow
   I0927 14:51:40.211920 31336 parse.cc:135] 562.249 ms loading parser
   I0927 14:51:40.211973 31336 parse.cc:194] Evaluating parser on dev.rec
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

