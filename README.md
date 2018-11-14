# SLING - A natural language frame semantics parser

[![Build Status](https://travis-ci.org/google/sling.svg?branch=master)](https://travis-ci.org/google/sling)

SLING is a parser for annotating text with frame semantic annotations. It is a general 
transition-based frame semantic parser using bi-directional LSTMs for input encoding 
and a Transition Based Recurrent Unit (TBRU) for output decoding. It is a jointly 
trained model using only the text tokens as input and the transition system has been 
designed to output frame graphs directly without any intervening symbolic representation.

![SLING neural network architecture.](./doc/report/network.svg)

The SLING framework includes an efficient and scalable frame store
implementation as well as a neural network JIT compiler for fast parsing at
runtime.

A more detailed description of the SLING parser can be found in this paper:

* Michael Ringgaard, Rahul Gupta, and Fernando C. N. Pereira. 2017.
  *SLING: A framework for frame semantic parsing*. http://arxiv.org/abs/1710.07032.

</span>

## More information ...

  * [Installation and building](doc/guide/install.md)
  * [Training a parser](doc/guide/training.md)
  * [Running the parser](doc/guide/parsing.md)
  * [Semantic frames](doc/guide/frames.md)
  * [SLING Python API](doc/guide/pyapi.md)
  * [Myelin neural network JIT compiler](doc/guide/myelin.md)
  * [Wikipedia and Wikidata processing](doc/guide/wikiflow.md)

## Credits

Original authors of the code in this package include:

*   Michael Ringgaard
*   Rahul Gupta

