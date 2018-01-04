# Myelin - Neural network JIT compiler

[![Build Status](https://travis-ci.org/google/sling.svg?branch=myelin)](https://travis-ci.org/google/sling)

Myelin is a just-in-time compiler for neural networks. It compiles a
_flow_ into x64 assembly code at runtime. The flow contains the graph for the
neural network computations as well as the learned weights from training the
network. The generated code takes the CPU features of the machine into account
when generating the code so it can take advantage of specialized features like
SSE, AVX, and FMA3.

See [here](https://github.com/google/sling/tree/master/sling/myelin) for
further information.

