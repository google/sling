package(default_visibility = ["//visibility:public"])

cc_library(
  name = "transition-generator",
  srcs = ["transition-generator.cc"],
  hdrs = ["transition-generator.h"],
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
    ":transition-generator",
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
    ":feature-extractor",
    ":shared-resources",
    "//base",
    "//file:posix",
    "//frame:object",
    "//frame:serialization",
    "//frame:store",
    "//nlp/document:document",
    "//nlp/document:document-source",
    "//nlp/document:features",
    "//string:strcat",
    "//third_party/syntaxnet",
    "//third_party/zlib:miniunzip",
    "//third_party/zlib:zlib",
  ],
)

cc_library(
  name = "sempar-instance",
  hdrs = ["sempar-instance.h"],
  deps = [
    "//frame:store",
    "//nlp/document:document",
    "//third_party/syntaxnet",
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
    "//third_party/syntaxnet",
  ],
)

cc_library(
  name = "transition-state",
  hdrs = ["transition-state.h"],
  srcs = ["transition-state.cc"],
  deps = [
    ":transition-generator",
    ":sempar-instance",
    ":shared-resources",
    ":transition-system-type",
    "//frame:store",
    "//nlp/document:document",
    "//nlp/document:features",
    "//nlp/parser:parser-action",
    "//nlp/parser:parser-state",
    "//nlp/parser:roles",
    "//third_party/syntaxnet",
  ],
)

cc_library(
  name = "shared-resources",
  hdrs = ["shared-resources.h"],
  srcs = ["shared-resources.cc"],
  deps = [
    "//file",
    "//frame:serialization",
    "//frame:store",
    "//nlp/document:lexicon",
    "//nlp/parser:action-table",
    "//nlp/parser:roles",
    "//third_party/syntaxnet:syntaxnet",
  ],
)

cc_library(
  name = "transition-system-type",
  hdrs = ["transition-system-type.h"],
)

cc_library(
  name = "feature-extractor",
  hdrs = ["feature-extractor.h"],
  srcs = ["feature-extractor.cc"],
  deps = [
    ":shared-resources",
    ":transition-state",
    "//base",
    "//nlp/document:features",
    "//nlp/document:lexicon",
    "//third_party/syntaxnet",
  ],
  alwayslink = 1,
)

cc_library(
  name = "sempar-component",
  hdrs = ["sempar-component.h"],
  srcs = ["sempar-component.cc"],
  deps = [
    ":document-batch",
    ":feature-extractor",
    ":sempar-instance",
    ":transition-state",
    "//file",
    "//frame:store",
    "//nlp/document:document",
    "//nlp/parser:action-table",
    "//nlp/parser:parser-action",
    "//nlp/parser:parser-state",
    "//third_party/syntaxnet",
  ],
  alwayslink = 1,
)

cc_binary(
  name = "sempar.so",
  deps = [
    ":sempar-component",
    "//base:libinit",
    "//file:posix",
    "//third_party/syntaxnet:dragnn-ops",
  ],
  linkshared = 1,
  linkstatic = 1,
)

cc_library(
  name = "frame-evaluation",
  hdrs = ["frame-evaluation.h"],
  srcs = ["frame-evaluation.cc"],
  deps = [
    "//base",
    "//file",
    "//frame:object",
    "//frame:serialization",
    "//frame:store",
    "//nlp/document:document",
    "//nlp/document:document-source",
    "//string:strcat",
  ],
)

cc_binary(
  name = "evaluate-frames",
  srcs = ["evaluate-frames.cc"],
  deps = [
    ":frame-evaluation",
    "//base",
    "//file:posix",
    "//third_party/zlib:miniunzip",
    "//third_party/zlib:zlib",
  ],
)

