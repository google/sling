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
      A tensor of scores, ordered by {batch_size, num_actions}.
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
_emit_all_final_outputs = ["all_final"]


def emit_all_final(handle, component, name=None):
  r"""Given a ComputeSession and Component, returns whether the Component is final.

  A component is considered final when all elements in the batch are final.

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
    A batch_size vector of gold labels for the current
    ComputeSession.
  """
  result = _op_def_lib.apply_op("EmitOracleLabels", handle=handle,
                                component=component, name=name)
  return result


_ops.RegisterShape("EmitOracleLabels")(None)
_extract_fixed_features_outputs = ["ids"]


_ExtractFixedFeaturesOutput = _collections.namedtuple("ExtractFixedFeatures",
                                                      _extract_fixed_features_outputs)


def extract_fixed_features(handle, batch_size, component,
                           channel_id, max_num_ids, name=None):
  r"""Given a ComputeSession, Component, and channel index, output fixed features.

  Fixed features returned as an 'ids' tensor, which specifies which rows
  should be looked up in the embedding matrix.

  Args:
    handle: A `Tensor` of type `string`. A handle to a ComputeSession.
    batch_size: an `int`. The current batch size.
    component: A `string`. The component name.
    channel_id: An `int`. The feature channel to extract features for.
    max_num_ids: An `int`. Maximum number of output feature ids per batch item.
    name: A name for the operation (optional).

  Returns:
    ids: A `Tensor` of type `int64`. The indices into embedding matrices for each feature.
  """
  result = _op_def_lib.apply_op("ExtractFixedFeatures", handle=handle,
                                batch_size=batch_size,
                                component=component, channel_id=channel_id,
                                max_num_ids=max_num_ids,
                                name=name)
  #return _ExtractFixedFeaturesOutput._make(result)
  return result


_ops.RegisterShape("ExtractFixedFeatures")(None)
_extract_link_features_outputs = ["step_idx", "idx"]


_ExtractLinkFeaturesOutput = _collections.namedtuple("ExtractLinkFeatures",
                                                     _extract_link_features_outputs)


def extract_link_features(handle, batch_size, component,
                          channel_id, channel_size, name=None):
  r"""Given a ComputeSession, Component, and a channel index, outputs link features.

  Output indices have shape {batch_size * channel_size}.

  Args:
    handle: A `Tensor` of type `string`. A handle to a ComputeSession.
    batch_size: an `int`. The current batch size.
    component: A `string`.
      The name of a Component instance, matching the ComponentSpec.name.
    channel_id: An `int`. The feature channel to extract features for.
    channel_size: An `int`. Channel size of channel_id.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (step_idx, idx).
    step_idx: A `Tensor` of type `int32`. The step indices to read activations from.
    idx: A `Tensor` of type `int32`. indices The index within a step to read the activations from.
  """
  result = _op_def_lib.apply_op("ExtractLinkFeatures", handle=handle,
                                batch_size=batch_size,
                                component=component, channel_id=channel_id,
                                channel_size=channel_size,
                                name=name)
  return _ExtractLinkFeaturesOutput._make(result)


_ops.RegisterShape("ExtractLinkFeatures")(None)
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


def init_component_data(
    handle, component, clear_existing_annotations, name=None):
  r"""Initialize a component for a given ComputeSession.

  Args:
    handle: A `Tensor` of type `string`. A handle to a ComputeSession.
    component: A `string`.
      The name of a Component instance, matching the ComponentSpec.name.
    clear_existing_annotations: Bool that says whether to clear existing
      annotations in the input documents.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
    The handle to the same ComputeSession after initialization.
  """
  result = _op_def_lib.apply_op(
      "InitComponentData", handle=handle,
      component=component,
      clear_existing_annotations=clear_existing_annotations,
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
  input_arg {
    name: "batch_size"
    type: DT_INT32
  }
  output_arg {
    name: "ids"
    type: DT_INT64
  }
  attr {
    name: "component"
    type: "string"
  }
  attr {
    name: "channel_id"
    type: "int"
  }
  attr {
    name: "max_num_ids"
    type: "int"
  }
}
op {
  name: "ExtractLinkFeatures"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  input_arg {
    name: "batch_size"
    type: DT_INT32
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
  attr {
    name: "channel_size"
    type: "int"
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
  output_arg {
    name: "output_handle"
    type: DT_STRING
  }
  attr {
    name: "component"
    type: "string"
  }
  attr {
    name: "clear_existing_annotations"
    type: "bool"
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
