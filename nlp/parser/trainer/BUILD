package(default_visibility = ["//visibility:public"])

cc_library(
    name = "gold-transition-generator",
    srcs = ["gold-transition-generator.cc"],
    hdrs = ["gold-transition-generator.h"],
    deps = [
        "//base",
        "//nlp/document",
        "//nlp/parser:parser-action",
        "//frame:object",
        "//frame:store",
        "//string:strcat",
    ],
)

cc_library(
    name = "action-table-generator",
    srcs = ["action-table-generator.cc"],
    hdrs = ["action-table-generator.h"],
    deps = [
        ":gold-transition-generator",
        "//base",
        "//nlp/parser:action-table",
        "//nlp/parser:parser-state",
        "//file",
        "//frame:object",
        "//frame:serialization",
        "//frame:store",
        "//nlp/document",
        "//string:strcat",
    ],
)

cc_binary(
  name = "generate-master-spec",
  srcs = ["generate-master-spec.cc"],
  deps = [
    ":action-table-generator",
    ":fixed-features",
    ":feature",
    ":link-features",
    ":shared-resources",
    "//base",
    "//file:file",
    "//file:posix",
    "//frame:object",
    "//frame:serialization",
    "//frame:store",
    "//nlp/document:document",
    "//nlp/document:document-source",
    "//string:strcat",
    "//third_party/sempar_dragnn:sempar_dragnn",
  ],
)

cc_library(
  name = "sempar-instance",
  hdrs = ["sempar-instance.h"],
  srcs = [],
  deps = [
      "//frame:store",
      "//nlp/document:document",
      "//third_party/sempar_dragnn:sempar_dragnn",
  ],
)

cc_library(
  name = "document-batch",
  hdrs = ["document-batch.h"],
  srcs = ["document-batch.cc"],
  deps = [
      ":sempar-instance",
      "//frame:object",
      "//frame:serialization",
      "//frame:store",
      "//nlp/document:document",
      "//third_party/sempar_dragnn:sempar_dragnn",
  ],
)

cc_library(
  name = "feature-extractor",
  hdrs = ["feature-extractor.h"],
  srcs = [],
  deps = [
      "//frame:object",
      "//frame:serialization",
      "//frame:store",
      "//nlp/document:document",
      "//third_party/sempar_dragnn:sempar_dragnn",
  ],
)

cc_library(
  name = "transition-state",
  hdrs = ["transition-state.h"],
  srcs = ["transition-state.cc"],
  deps = [
      ":gold-transition-generator",
      ":sempar-instance",
      ":shared-resources",
      ":transition-system-type",
      "//frame:store",
      "//nlp/document:document",
      "//nlp/parser:parser-action",
      "//nlp/parser:parser-state",
      "//third_party/sempar_dragnn:sempar_dragnn",
  ],
)

cc_library(
  name = "workspace",
  hdrs = ["workspace.h"],
  srcs = [],
  deps = [
      "//base",
  ],
)

cc_library(
  name = "shared-resources",
  hdrs = ["shared-resources.h"],
  srcs = ["shared-resources.cc"],
  deps = [
      "//frame:serialization",
      "//frame:store",
      "//nlp/parser:action-table",
  ],
)

cc_library(
  name = "transition-system-type",
  hdrs = ["transition-system-type.h"],
  srcs = [],
  deps = [],
)

cc_library(
  name = "feature",
  hdrs = ["feature.h"],
  srcs = ["feature.cc"],
  deps = [
      ":shared-resources",
      ":transition-state",
      "//frame:serialization",
      "//nlp/document:document-source",
      "//string:numbers",
      "//third_party/sempar_dragnn:sempar_dragnn",
  ],
  alwayslink = 1,
)

cc_library(
  name = "fixed-features",
  hdrs = [],
  srcs = ["fixed-features.cc"],
  deps = [
      ":feature",
      ":shared-resources",
      "//file",
      "//stream:file-input",
      "//string:strcat",
      "//third_party/sempar_dragnn:sempar_dragnn",
  ],
  alwayslink = 1,
)

cc_library(
  name = "link-features",
  hdrs = [],
  srcs = ["link-features.cc"],
  deps = [
      ":feature",
      "//string:strcat",
      "//third_party/sempar_dragnn:sempar_dragnn",
  ],
  alwayslink = 1,
)

cc_library(
  name = "sempar-component",
  hdrs = ["sempar-component.h"],
  srcs = ["sempar-component.cc"],
  deps = [
      ":document-batch",
      ":feature",
      ":sempar-instance",
      ":transition-state",
      "//frame:store",
      "//nlp/document:document",
      "//nlp/parser:action-table",
      "//nlp/parser:parser-action",
      "//nlp/parser:parser-state",
      "//third_party/sempar_dragnn:sempar_dragnn",
  ],
  alwayslink = 1,
)

cc_binary(
  name = "sempar-component-test",
  srcs = ["sempar-component-test.cc"],
  deps = [
    ":fixed-features",
    ":feature",
    ":link-features",
    ":sempar-component",
    ":shared-resources",
    "//base",
    "//file:file",
    "//file:posix",
    "//frame:object",
    "//frame:serialization",
    "//frame:store",
    "//nlp/parser:parser-action",
    "//string:strcat",
    "//third_party/sempar_dragnn:sempar_dragnn",
  ],
)

py_binary(
    name = "graph-builder-main",
    srcs = ["graph-builder-main.py"],
    data = [],
    deps = [],
)

cc_library(
  name = "frame-evaluation-lib",
  hdrs = ["frame-evaluation.h"],
  srcs = ["frame-evaluation.cc"],
  deps = [
    "//base",
    "//file:file",
    "//file:posix",
    "//frame:object",
    "//frame:serialization",
    "//frame:store",
    "//nlp/document:document",
    "//string:strcat",
  ],
)

cc_binary(
  name = "frame-evaluation",
  srcs = ["frame-evaluation-main.cc"],
  deps = [
    ":frame-evaluation-lib",
    "//base",
    "//file:file",
    "//file:posix",
  ],
)

