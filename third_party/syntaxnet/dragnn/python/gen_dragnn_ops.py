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
_advance_from_oracle_outputs = ["output_handle"]


def advance_from_oracle(handle, component, name=None):
  r"""Given a ComputeSession and a Component name, advance the component via oracle.

  Args:
    handle: A `Tensor` of type `string`. A handle to a ComputeSession.
    component: A `string`.
      The name of a Component instance, matching the ComponentSpec.name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
    The handle to the same ComputeSession after advancement.
  """
  result = _op_def_lib.apply_op("AdvanceFromOracle", handle=handle,
                                component=component, name=name)
  return result


_ops.RegisterShape("AdvanceFromOracle")(None)
_advance_from_prediction_outputs = ["output_handle"]


def advance_from_prediction(handle, scores, component, name=None):
  r"""Given a ComputeSession, a Component name, and a score tensor, advance the state.

  Args:
    handle: A `Tensor` of type `string`. A handle to a ComputeSession.
    scores: A `Tensor` of type `float32`.
      A tensor of scores, ordered by {batch_size, beam_size, num_actions}.
    component: A `string`.
      The name of a Component instance, matching the ComponentSpec.name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
    A handle to the same ComputeSession after advancement.
  """
  result = _op_def_lib.apply_op("AdvanceFromPrediction", handle=handle,
                                scores=scores, component=component, name=name)
  return result


_ops.RegisterShape("AdvanceFromPrediction")(None)
_attach_data_reader_outputs = ["output_handle"]


def attach_data_reader(handle, input_spec, component=None, name=None):
  r"""Given a ComputeSession, attach a data source.

  This op is agnostic to the type of input data. The vector of input strings is
  interpreted by the backend.

  Args:
    handle: A `Tensor` of type `string`. A handle to a ComputeSession.
    input_spec: A `Tensor` of type `string`.
      A vector of strings, where each string represents one batch item.
    component: An optional `string`. Defaults to `"NOT_USED_FOR_THIS_OP"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
    The handle to the same ComputeSession after attachment.
  """
  result = _op_def_lib.apply_op("AttachDataReader", handle=handle,
                                input_spec=input_spec, component=component,
                                name=name)
  return result


_ops.RegisterShape("AttachDataReader")(None)
_batch_size_outputs = ["batch_size"]


def batch_size(handle, component, name=None):
  r"""Given a ComputeSession and a component name,return the component batch size.

  Args:
    handle: A `Tensor` of type `string`. A handle to a ComputeSession.
    component: A `string`.
      The name of a Component instance, matching the ComponentSpec.name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`. The size of the given component's batch.
  """
  result = _op_def_lib.apply_op("BatchSize", handle=handle,
                                component=component, name=name)
  return result


_ops.RegisterShape("BatchSize")(None)
_dragnn_embedding_initializer_outputs = ["embeddings"]


def dragnn_embedding_initializer(embedding_input, vocab,
                                 scaling_coefficient=None, seed=None,
                                 seed2=None, name=None):
  r"""*** PLACEHOLDER OP - FUNCTIONALITY NOT YET IMPLEMENTED ***

  Read embeddings from an an input for every key specified in a text vocab file.

  Args:
    embedding_input: A `string`. Path to location with embedding vectors.
    vocab: A `string`. Path to list of keys corresponding to the input.
    scaling_coefficient: An optional `float`. Defaults to `1`.
      A scaling coefficient for the embedding matrix.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
    A tensor containing embeddings from the specified sstable.
  """
  result = _op_def_lib.apply_op("DragnnEmbeddingInitializer",
                                embedding_input=embedding_input, vocab=vocab,
                                scaling_coefficient=scaling_coefficient,
                                seed=seed, seed2=seed2, name=name)
  return result


_ops.RegisterShape("DragnnEmbeddingInitializer")(None)
_emit_all_final_outputs = ["all_final"]


def emit_all_final(handle, component, name=None):
  r"""Given a ComputeSession and Component, returns whether the Component is final.

  A component is considered final when all elements in the batch have beams
  containing all final states.

  Args:
    handle: A `Tensor` of type `string`. A handle to a ComputeSession.
    component: A `string`.
      The name of a Component instance, matching the ComponentSpec.name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
    Whether every element in the specified component is 'final'.
  """
  result = _op_def_lib.apply_op("EmitAllFinal", handle=handle,
                                component=component, name=name)
  return result


_ops.RegisterShape("EmitAllFinal")(None)
_emit_annotations_outputs = ["annotations"]


def emit_annotations(handle, component, name=None):
  r"""Given a ComputeSession, emits strings with final predictions for the model.

  Predictions are given for each element in the final component's batch.

  Args:
    handle: A `Tensor` of type `string`. A handle to a ComputeSession.
    component: A `string`.
      The name of a Component instance, matching the ComponentSpec.name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
    A vector of strings representing the annotated data.
  """
  result = _op_def_lib.apply_op("EmitAnnotations", handle=handle,
                                component=component, name=name)
  return result


_ops.RegisterShape("EmitAnnotations")(None)
_emit_oracle_labels_outputs = ["gold_labels"]


def emit_oracle_labels(handle, component, name=None):
  r"""Given a ComputeSession and Component, emit a vector of gold labels.

  Args:
    handle: A `Tensor` of type `string`. A handle to a ComputeSession.
    component: A `string`.
      The name of a Component instance, matching the ComponentSpec.name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
    A [batch_size * beam_size] vector of gold labels for the current
    ComputeSession.
  """
  result = _op_def_lib.apply_op("EmitOracleLabels", handle=handle,
                                component=component, name=name)
  return result


_ops.RegisterShape("EmitOracleLabels")(None)
_extract_fixed_features_outputs = ["indices", "ids", "weights"]


_ExtractFixedFeaturesOutput = _collections.namedtuple("ExtractFixedFeatures",
                                                      _extract_fixed_features_outputs)


def extract_fixed_features(handle, component, channel_id, name=None):
  r"""Given a ComputeSession, Component, and channel index, output fixed features.

  Fixed features returned as 3 vectors, 'indices', 'ids', and 'weights' of equal
  length. 'ids' specifies which rows should be looked up in the embedding
  matrix. 'weights' specifies a scale for each embedding vector. 'indices' is a
  sorted vector that assigns the same index to embedding vectors that should be
  summed together.

  Args:
    handle: A `Tensor` of type `string`. A handle to a ComputeSession.
    component: A `string`.
      The name of a Component instance, matching the ComponentSpec.name.
    channel_id: An `int`. The feature channel to extract features for.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (indices, ids, weights).
    indices: A `Tensor` of type `int32`. The row to add the feature to.
    ids: A `Tensor` of type `int64`. The indices into embedding matrices for each feature.
    weights: A `Tensor` of type `float32`. The weight for each looked up feature.
  """
  result = _op_def_lib.apply_op("ExtractFixedFeatures", handle=handle,
                                component=component, channel_id=channel_id,
                                name=name)
  return _ExtractFixedFeaturesOutput._make(result)


_ops.RegisterShape("ExtractFixedFeatures")(None)
_extract_link_features_outputs = ["step_idx", "idx"]


_ExtractLinkFeaturesOutput = _collections.namedtuple("ExtractLinkFeatures",
                                                     _extract_link_features_outputs)


def extract_link_features(handle, component, channel_id, name=None):
  r"""Given a ComputeSession, Component, and a channel index, outputs link features.

  Output indices have shape {batch_size * beam_size * channel_size}.

  Args:
    handle: A `Tensor` of type `string`. A handle to a ComputeSession.
    component: A `string`.
      The name of a Component instance, matching the ComponentSpec.name.
    channel_id: An `int`. The feature channel to extract features for.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (step_idx, idx).
    step_idx: A `Tensor` of type `int32`. The step indices to read activations from.
    idx: A `Tensor` of type `int32`. indices The index within a step to read the activations from.
  """
  result = _op_def_lib.apply_op("ExtractLinkFeatures", handle=handle,
                                component=component, channel_id=channel_id,
                                name=name)
  return _ExtractLinkFeaturesOutput._make(result)


_ops.RegisterShape("ExtractLinkFeatures")(None)
_get_component_trace_outputs = ["trace"]


def get_component_trace(handle, component, name=None):
  r"""Gets the raw MasterTrace proto for each batch, state, and beam slot.

  Args:
    handle: A `Tensor` of type `string`. A handle to a ComputeSession.
    component: A `string`.
      The name of a Component instance, matching the ComponentSpec.name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. A vector of MasterTrace protos.
  """
  result = _op_def_lib.apply_op("GetComponentTrace", handle=handle,
                                component=component, name=name)
  return result


_ops.RegisterShape("GetComponentTrace")(None)
_get_session_outputs = ["handle"]


def get_session(container, master_spec, grid_point, name=None):
  r"""Given MasterSpec and GridPoint protos, outputs a handle to a ComputeSession.

  Args:
    container: A `Tensor` of type `string`.
      A unique identifier for the ComputeSessionPool from which a
      ComputeSession will be allocated.
    master_spec: A `string`. A serialized syntaxnet.dragnn.MasterSpec proto.
    grid_point: A `string`. A serialized syntaxnet.dragnn.GridPoint proto.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. A string handle to a ComputeSession.
  """
  result = _op_def_lib.apply_op("GetSession", container=container,
                                master_spec=master_spec,
                                grid_point=grid_point, name=name)
  return result


_ops.RegisterShape("GetSession")(None)
_init_component_data_outputs = ["output_handle"]


def init_component_data(handle, beam_size, component, name=None):
  r"""Initialize a component with the given beam size for a given ComputeSession.

  Args:
    handle: A `Tensor` of type `string`. A handle to a ComputeSession.
    beam_size: A `Tensor` of type `int32`.
      The size of the beam to use on the component.
    component: A `string`.
      The name of a Component instance, matching the ComponentSpec.name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
    The handle to the same ComputeSession after initialization.
  """
  result = _op_def_lib.apply_op("InitComponentData", handle=handle,
                                beam_size=beam_size, component=component,
                                name=name)
  return result


_ops.RegisterShape("InitComponentData")(None)
_release_session_outputs = [""]


def release_session(handle, name=None):
  r"""Given a ComputeSession, return it to the ComputeSession pool.

  This ComputeSession will no longer be available after this op returns.

  Args:
    handle: A `Tensor` of type `string`.
      A handle to a ComputeSession that will be returned to the backing pool.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ReleaseSession", handle=handle, name=name)
  return result


_ops.RegisterShape("ReleaseSession")(None)
_set_tracing_outputs = ["output_handle"]


def set_tracing(handle, tracing_on, component=None, name=None):
  r"""Given a ComputeSession, turns on or off tracing for all components.

  Args:
    handle: A `Tensor` of type `string`. A handle to a ComputeSession.
    tracing_on: A `Tensor` of type `bool`. Whether or not to record traces.
    component: An optional `string`. Defaults to `"NOT_USED_FOR_THIS_OP"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
    The handle to the same ComputeSession, with the tracing status changed.
  """
  result = _op_def_lib.apply_op("SetTracing", handle=handle,
                                tracing_on=tracing_on, component=component,
                                name=name)
  return result


_ops.RegisterShape("SetTracing")(None)
_write_annotations_outputs = ["output_handle"]


def write_annotations(handle, component, name=None):
  r"""Given a ComputeSession, has the given component write out its annotations.

  The annotations are written to the underlying data objects passed in at the
  beginning of the computation.

  Args:
    handle: A `Tensor` of type `string`. A handle to a ComputeSession.
    component: A `string`.
      The name of a Component instance, matching the ComponentSpec.name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
    A handle to the same ComputeSession after writing.
  """
  result = _op_def_lib.apply_op("WriteAnnotations", handle=handle,
                                component=component, name=name)
  return result


_ops.RegisterShape("WriteAnnotations")(None)
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "AdvanceFromOracle"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  output_arg {
    name: "output_handle"
    type: DT_STRING
  }
  attr {
    name: "component"
    type: "string"
  }
}
op {
  name: "AdvanceFromPrediction"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  input_arg {
    name: "scores"
    type: DT_FLOAT
  }
  output_arg {
    name: "output_handle"
    type: DT_STRING
  }
  attr {
    name: "component"
    type: "string"
  }
}
op {
  name: "AttachDataReader"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  input_arg {
    name: "input_spec"
    type: DT_STRING
  }
  output_arg {
    name: "output_handle"
    type: DT_STRING
  }
  attr {
    name: "component"
    type: "string"
    default_value {
      s: "NOT_USED_FOR_THIS_OP"
    }
  }
}
op {
  name: "BatchSize"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  output_arg {
    name: "batch_size"
    type: DT_INT32
  }
  attr {
    name: "component"
    type: "string"
  }
}
op {
  name: "DragnnEmbeddingInitializer"
  output_arg {
    name: "embeddings"
    type: DT_FLOAT
  }
  attr {
    name: "embedding_input"
    type: "string"
  }
  attr {
    name: "vocab"
    type: "string"
  }
  attr {
    name: "scaling_coefficient"
    type: "float"
    default_value {
      f: 1
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
op {
  name: "EmitAllFinal"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  output_arg {
    name: "all_final"
    type: DT_BOOL
  }
  attr {
    name: "component"
    type: "string"
  }
}
op {
  name: "EmitAnnotations"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  output_arg {
    name: "annotations"
    type: DT_STRING
  }
  attr {
    name: "component"
    type: "string"
  }
}
op {
  name: "EmitOracleLabels"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  output_arg {
    name: "gold_labels"
    type: DT_INT32
  }
  attr {
    name: "component"
    type: "string"
  }
}
op {
  name: "ExtractFixedFeatures"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  output_arg {
    name: "indices"
    type: DT_INT32
  }
  output_arg {
    name: "ids"
    type: DT_INT64
  }
  output_arg {
    name: "weights"
    type: DT_FLOAT
  }
  attr {
    name: "component"
    type: "string"
  }
  attr {
    name: "channel_id"
    type: "int"
  }
}
op {
  name: "ExtractLinkFeatures"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  output_arg {
    name: "step_idx"
    type: DT_INT32
  }
  output_arg {
    name: "idx"
    type: DT_INT32
  }
  attr {
    name: "component"
    type: "string"
  }
  attr {
    name: "channel_id"
    type: "int"
  }
}
op {
  name: "GetComponentTrace"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  output_arg {
    name: "trace"
    type: DT_STRING
  }
  attr {
    name: "component"
    type: "string"
  }
}
op {
  name: "GetSession"
  input_arg {
    name: "container"
    type: DT_STRING
  }
  output_arg {
    name: "handle"
    type: DT_STRING
  }
  attr {
    name: "master_spec"
    type: "string"
  }
  attr {
    name: "grid_point"
    type: "string"
  }
  is_stateful: true
}
op {
  name: "InitComponentData"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  input_arg {
    name: "beam_size"
    type: DT_INT32
  }
  output_arg {
    name: "output_handle"
    type: DT_STRING
  }
  attr {
    name: "component"
    type: "string"
  }
}
op {
  name: "ReleaseSession"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  is_stateful: true
}
op {
  name: "SetTracing"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  input_arg {
    name: "tracing_on"
    type: DT_BOOL
  }
  output_arg {
    name: "output_handle"
    type: DT_STRING
  }
  attr {
    name: "component"
    type: "string"
    default_value {
      s: "NOT_USED_FOR_THIS_OP"
    }
  }
}
op {
  name: "WriteAnnotations"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  output_arg {
    name: "output_handle"
    type: DT_STRING
  }
  attr {
    name: "component"
    type: "string"
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
