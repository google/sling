"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections as _collections

from google.protobuf import text_format as _text_format

from tensorflow.core.framework import op_def_pb2 as _op_def_pb2

# Needed to trigger the call to _set_call_cpp_shape_fn.
from tensorflow.python.framework import common_shapes as _common_shapes

from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library

_word_embedding_initializer_outputs = ["word_embeddings"]


def word_embedding_initializer(vectors, vocabulary=None,
                               seed=None, seed2=None,
                               name=None):
  r"""Reads word embeddings from an sstable of dist_belief.TokenEmbedding protos for

  every word specified in a text vocabulary file.

  Args:
    vectors: A `string`. path to TF record file of word embedding vectors.
    vocabulary: An optional `string`. Defaults to `""`.
      path to vocabulary file, which contains one unique word per line.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a random
      seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
    a tensor containing word embeddings from the specified table.
  """
  result = _op_def_lib.apply_op("WordEmbeddingInitializer", vectors=vectors,
                                vocabulary=vocabulary,
                                seed=seed,
                                seed2=seed2, name=name)
  return result


_ops.RegisterShape("WordEmbeddingInitializer")(None)
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """
op {
  name: "WordEmbeddingInitializer"
  output_arg {
    name: "word_embeddings"
    type: DT_FLOAT
  }
  attr {
    name: "vectors"
    type: "string"
  }
  attr {
    name: "vocabulary"
    type: "string"
    default_value {
      s: ""
    }
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
