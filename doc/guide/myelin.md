# Myelin - Neural network JIT compiler

Myelin is a just-in-time compiler for neural networks. It compiles a
_flow_ into x64 assembly code at runtime. The flow contains the graph for the
neural network computations as well as the learned weights from training the
network. The generated code takes the CPU features of the machine into account
when generating the code so it can take advantage of specialized features like
SSE, AVX, and FMA3.

Myelin can be used at inference time (as opposed to training time) to speed up
neural network computations. The neural network is stored in a _.flow_ file
which is loaded and compiled into a _network_ at runtime by Myelin.

## Platform

Operating system: Linux<br>
Languages: C++, assembler, Python<br>
CPU: Intel x64 or compatible<br>
Build system: Bazel<br>

## Creating flow files

Myelin uses [flow files](#flow-file-format) to store neural networks. A
Tensorflow graph can be stored as a flow file using the myelin Python module.
After the network has been trained, the parts of the Tensorflow graph needed
for inference can be exported to a flow file. The following is a simple example
of training a MNIST classifier and storing the resulting network in a flow
file:

```python
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from sling.myelin import Flow
from sling.myelin import Builder

# Import data.
mnist = input_data.read_data_sets("/tmp/mnist", one_hot=True)

# Create the model.
x = tf.placeholder(tf.float32, [None, 784], name='x')
W = tf.Variable(tf.zeros([784, 10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')
y = tf.add(tf.matmul(x, W), b, name='y')

# Define loss and optimizer.
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Train model.
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Save model to flow file.
flow = Flow()
builder = Builder(sess, flow)
builder.add(flow.func("classifier"), [x], [y])
flow.save("/tmp/mnist.flow")
```

This will extract the parts of the TF graph needed for computing `y` from `x`.
It will add a _function_ (`classifier`) to the flow and then add the `MatMul`
and `Add` _operations_ to this function. It will also add `W` and `b` as
constant _variables_ to the flow with the trained weights. The resulting flow
is then saved to the file _/tmp/model.flow_.

If the Tensorflow graph has been saved to a checkpoint using a TF Saver object,
you can load the checkpoint and only store the parts needed for inference as
a flow file:

```python
import tensorflow as tf
from sling.myelin import Flow
from sling.myelin import Builder

# Load Tensorflow checkpoint.
sess = tf.Session()
saver = tf.train.import_meta_graph('/tmp/mnist.ckpt.meta')
saver.restore(sess, '/tmp/mnist.ckpt')

# Create Myelin flow.
flow = Flow()
builder = Builder(sess, flow)

# Extract flow from graph.
inputs = [sess.graph.get_tensor_by_name("x:0")]
outputs = [sess.graph.get_tensor_by_name("y:0")]
builder.add(flow.func("classifier"), inputs, outputs)

# Save flow.
flow.save("/tmp/mnist.flow")
```

## Setting up a kernel library

```c++
#include "sling/myelin/compute.h"
#include "sling/myelin/kernel/tensorflow.h"

using namespace sling::myelin;

// Initialize library with kernels.
Library library;
RegisterTensorflowLibrary(&library);
```

Myelin uses a library of transformations and kernels for generating code
for the neural network. In this example we add generic kernels which can be
used on any x64 processor as well as specialized kernels for CPUs with
[AVX](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) support. You can
add your own kernel generators and graph transformations for custom ops or
for generating optimized code for special cases of standard ops.

## Compiling a network

```c++
// Load and compile neural network.
Network nn;
CHECK(nn.Compile("/tmp/mnist.flow", library));

// Get neural network cell for classifier.
Cell *classifier = nn.GetCell("classifier");

// Get classifier inputs and outputs.
Tensor *x = classifier->GetParameter("x:0");
Tensor *y = classifier->GetParameter("y:0");
```

Myelin can load and compile a flow file into a `Network` object. This contains
a _cell_ for for each function in the flow file. A `Cell` holds the generated
code for computing the cell function as well as a description of the data layout
for all the parameters used by the computation. The parameters can be input
(e.g. `x`), output parameters (e.g. `y`), or intermediate values needed by the
cell computation. Constant parameters (e.g. `W` and `b`) are stored in the
`Network` object and can be shared between cells.

After the network has been compiled, the parameters can be looked up in the
cell or network. The `Tensor` object then knows the location of the parameter
in the compiled flow.

## Computing cell functions

```c++
// Create instance of neural network cell for classifying input.
Instance data(classifier);

// Set input for classification.
float *input = data.Get<float>(x);
<<< fill input array with image data >>>

// Classify input.
data.Compute();

// Get output prediction.
float *output = data.Get<float>(y);
<<< argmax of output distribution is the prediction >>
```

An `Instance` object is used for computing a cell function. The instance
allocates memory for all the input, output, and intermediate parameters for
the cell. Multiple instance objects can be created for a cell function and
computations on these different instances can be done concurrently.

When an instance for a cell has been allocated, the input parameters should be
filled into the instance. The `Compute()` method then invokes the generated code
for the cell and computes the output parameters from the input parameters (and
constant parameters). Then the values of the output parameters can be read from
the instance object.

An instance object can be used for multiple computations, but the `Clear()`
method needs to be called before the instance can be reused for another
computation.

## Flow file format

A flow file contains a trained neural network with variables, operations,
functions, connectors, and blobs. It is a simple binary file format with the
following structure:

```
flow = "flow" <version>
       <#flags> (unused, from version 5)
       <#vars> var*
       <#ops> op*
       <#funcs> func*
       <#cnxs> cnx*
       <#blobs> blob* (from version 4)

var = <name$>
      <#flags> (IN=1, OUT=2, REF=4, LEARNABLE=8 UNIQUE=16, from version 5)
      <#aliases> <alias$>
      <dtype$>
      <shape>
      <#bytes> value

op = <name$>
     <#flags> (unused, from version 5)
     <type$>
     <#inputs> <input$>*
     <#outputs> <output$>*
     <#attrs> attr*

blob = <name$>
       <#flags> (unused, from version 5)
       <type$>
       <#attrs> attr*
       <#bytes> data

func = <name$>
       <#flags> (TRAINING=1, from version 5)
       <#ops> <op$>

cnx = <name$>
      <#flags> (unused, from version 5)
      <#vars> <var$>

shape = <#dims> <size>*

attr = <name$> <value$>

dtype = "float16" | "float32" | "float64" | "int8" | "uint8" |
        "int16" | "uint16" | "int32" | "uint64"

"flow" = 0x776f6c66
version = 3 | 4 | 5
```

A flow file begins with the _magic_ string "flow" followed by a version number.
Numbers are encoded as 32-bit integers stored in little-endian format (aka Intel
format). Strings are stored as length-prefixed strings where the length is
encoded as a 32-bit integer. Constant data for variables are stored in numpy
ndarray row-major format with an unsigned 64-bit little-endian length prefix. If
a variable does not have any constant value, the length is zero.

