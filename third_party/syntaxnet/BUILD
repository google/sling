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
  name = "trace-proto",
  srcs = ["dragnn/protos/trace.proto"],
  deps = [":data-proto"],
)

proto_library(
  name = "dictionary-proto",
  srcs = ["syntaxnet/dictionary.proto"],
)

proto_library(
  name = "feature-extractor-proto",
  srcs = ["syntaxnet/feature_extractor.proto"],
)

proto_library(
  name = "sentence-proto",
  srcs = ["syntaxnet/sentence.proto"],
)

proto_library(
  name = "task-spec-proto",
  srcs = ["syntaxnet/task_spec.proto"],
)

proto_library(
  name = "sparse-proto",
  srcs = ["syntaxnet/sparse.proto"],
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
  name = "trace",
  deps = ["trace-proto"],
)

cc_proto_library(
  name = "dictionary",
  deps = ["dictionary-proto"],
)

cc_proto_library(
  name = "feature-extractor",
  deps = ["feature-extractor-proto"],
)

cc_proto_library(
  name = "sentence",
  deps = ["sentence-proto"],
)

cc_proto_library(
  name = "task-spec",
  deps = ["task-spec-proto"],
)

cc_proto_library(
  name = "sparse",
  deps = ["sparse-proto"],
)

cc_library(
  name = "syntaxnet",
  srcs = [
    "dragnn/core/component_registry.cc",
    "dragnn/io/sentence_input_batch.cc",
    "syntaxnet/affix.cc",
    "syntaxnet/char_properties.cc",
    "syntaxnet/document_format.cc",
    "syntaxnet/feature_extractor.cc",
    "syntaxnet/fml_parser.cc",
    "syntaxnet/registry.cc",
    "syntaxnet/shared_store.cc",
    "syntaxnet/task_context.cc",
    "syntaxnet/term_frequency_map.cc",
    "syntaxnet/utils.cc",
    "syntaxnet/workspace.cc",
    "util/utf8/rune.c",
    "util/utf8/unicodetext.cc",
    "util/utf8/unilib.cc",
  ],
  hdrs = [
    "dragnn/components/util/bulk_feature_extractor.h",
    "dragnn/core/component_registry.h",
    "dragnn/core/input_batch_cache.h",
    "dragnn/core/interfaces/cloneable_transition_state.h",
    "dragnn/core/interfaces/component.h",
    "dragnn/core/interfaces/input_batch.h",
    "dragnn/core/interfaces/transition_state.h",
    "dragnn/io/sentence_input_batch.h",
    "dragnn/io/syntaxnet_sentence.h",
    "syntaxnet/affix.h",
    "syntaxnet/base.h",
    "syntaxnet/char_properties.h",
    "syntaxnet/document_format.h",
    "syntaxnet/feature_extractor.h",
    "syntaxnet/feature_types.h",
    "syntaxnet/fml_parser.h",
    "syntaxnet/proto_io.h",
    "syntaxnet/task_context.h",
    "syntaxnet/registry.h",
    "syntaxnet/shared_store.h",
    "syntaxnet/term_frequency_map.h",
    "syntaxnet/utils.h",
    "syntaxnet/workspace.h",
    "util/utf8/unicodetext.h",
    "util/utf8/unilib.h",
    "util/utf8/unilib_utf8_utils.h",
    "util/utf8/utf.h",
    "util/utf8/utfdef.h",
  ],
  deps = [
    ":data",
    ":dictionary",
    ":feature-extractor",
    ":sentence",
    ":sparse",
    ":spec",
    ":task-spec",
    ":trace",
    "//third_party/tensorflow",
  ],
  includes = ["."],
  alwayslink = 1,
)

cc_library(
  name = "dragnn-ops",
  srcs = [
    "dragnn/components/stateless/stateless_component.cc",
    "dragnn/core/reader_ops.cc",
    "dragnn/core/compute_session_pool.cc",
    "dragnn/core/compute_session_impl.cc",
    "dragnn/core/index_translator.cc",
    "dragnn/core/ops/compute_session_op.cc",
    "dragnn/core/ops/dragnn_ops.cc",
    "dragnn/core/ops/dragnn_op_kernels.cc",
    "dragnn/core/ops/embedding_ops.cc",
  ],
  hdrs = [
    "dragnn/core/compute_session_impl.h",
    "dragnn/core/compute_session_pool.h",
    "dragnn/core/index_translator.h",
    "dragnn/core/resource_container.h",
    "dragnn/core/ops/compute_session_op.h",
  ],
  deps = [
    ":syntaxnet",
    "//third_party/tensorflow",
  ],
  alwayslink = 1,
)

