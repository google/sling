# SyntaxNet/DRAGNN

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

proto_library(
  name = "data-proto",
  srcs = ["dragnn/protos/data.proto"],
)

proto_library(
  name = "spec-proto",
  srcs = ["dragnn/protos/spec.proto"],
)

proto_library(
  name = "embedding-proto",
  srcs = ["dragnn/protos/embedding.proto"],
)

cc_proto_library(
  name = "data",
  deps = ["data-proto"],
)

cc_proto_library(
  name = "spec",
  deps = ["spec-proto"],
)

cc_proto_library(
  name = "embedding",
  deps = ["embedding-proto"],
)

cc_library(
  name = "syntaxnet",
  srcs = [
    "dragnn/core/component_registry.cc",
  ],
  hdrs = [
    "dragnn/core/component_registry.h",
    "dragnn/core/input_batch_cache.h",
    "dragnn/core/interfaces/component.h",
    "dragnn/core/interfaces/input_batch.h",
    "dragnn/core/interfaces/transition_state.h",
    "dragnn/core/proto_io.h",
  ],
  deps = [
    ":data",
    ":embedding",
    ":spec",
    "//base",
    "//base:registry",
    "//third_party/tensorflow",
  ],
  includes = ["."],
)

cc_library(
  name = "dragnn-ops",
  srcs = [
    "dragnn/core/reader_ops.cc",
    "dragnn/core/compute_session_pool.cc",
    "dragnn/core/compute_session.cc",
    "dragnn/core/index_translator.cc",
    "dragnn/core/ops/compute_session_op.cc",
    "dragnn/core/ops/dragnn_ops.cc",
    "dragnn/core/ops/dragnn_op_kernels.cc",
    "dragnn/core/ops/embedding_ops.cc",
  ],
  hdrs = [
    "dragnn/core/compute_session.h",
    "dragnn/core/compute_session_pool.h",
    "dragnn/core/index_translator.h",
    "dragnn/core/resource_container.h",
    "dragnn/core/ops/compute_session_op.h",
  ],
  deps = [
    ":syntaxnet",
    "//base",
    "//third_party/tensorflow",
  ],
  alwayslink = 1,
)

