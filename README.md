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

TDB...

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



