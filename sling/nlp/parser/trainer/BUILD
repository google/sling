package(default_visibility = ["//visibility:public"])

cc_library(
  name = "transition-generator",
  srcs = ["transition-generator.cc"],
  hdrs = ["transition-generator.h"],
  deps = [
    "//sling/base",
    "//sling/nlp/document",
    "//sling/nlp/parser:parser-action",
    "//sling/frame:object",
    "//sling/frame:store",
    "//sling/string:strcat",
  ],
)

cc_library(
  name = "action-table-generator",
  srcs = ["action-table-generator.cc"],
  hdrs = ["action-table-generator.h"],
  deps = [
    ":transition-generator",
    "//sling/base",
    "//sling/nlp/parser:action-table",
    "//sling/nlp/parser:parser-state",
    "//sling/file",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/nlp/document",
    "//sling/string:strcat",
  ],
)

cc_binary(
  name = "generate-master-spec",
  srcs = ["generate-master-spec.cc"],
  deps = [
    ":action-table-generator",
    ":feature-extractor",
    ":shared-resources",
    "//sling/base",
    "//sling/file:posix",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/nlp/document:document",
    "//sling/nlp/document:document-source",
    "//sling/nlp/document:features",
    "//sling/string:strcat",
    "//sling/util:embeddings",
    "//third_party/syntaxnet",
    "//third_party/zlib:miniunzip",
    "//third_party/zlib:zlib",
  ],
)

cc_library(
  name = "sempar-instance",
  hdrs = ["sempar-instance.h"],
  deps = [
    "//sling/frame:store",
    "//sling/nlp/document:document",
    "//third_party/syntaxnet",
  ],
)

cc_library(
  name = "document-batch",
  hdrs = ["document-batch.h"],
  srcs = ["document-batch.cc"],
  deps = [
    ":sempar-instance",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/nlp/document:document",
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
    "//sling/frame:store",
    "//sling/nlp/document:document",
    "//sling/nlp/document:features",
    "//sling/nlp/parser:parser-action",
    "//sling/nlp/parser:parser-state",
    "//sling/nlp/parser:roles",
    "//third_party/syntaxnet",
  ],
)

cc_library(
  name = "shared-resources",
  hdrs = ["shared-resources.h"],
  srcs = ["shared-resources.cc"],
  deps = [
    "//sling/file",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/nlp/document:lexicon",
    "//sling/nlp/parser:action-table",
    "//sling/nlp/parser:roles",
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
    "//sling/base",
    "//sling/nlp/document:features",
    "//sling/nlp/document:lexicon",
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
    "//sling/file",
    "//sling/frame:store",
    "//sling/nlp/document:document",
    "//sling/nlp/parser:action-table",
    "//sling/nlp/parser:parser-action",
    "//sling/nlp/parser:parser-state",
    "//third_party/syntaxnet",
  ],
  alwayslink = 1,
)

cc_binary(
  name = "sempar.so",
  deps = [
    ":sempar-component",
    "//sling/base:libinit",
    "//sling/file:posix",
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
    "//sling/base",
    "//sling/file",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/nlp/document:document",
    "//sling/nlp/document:document-source",
    "//sling/string:strcat",
  ],
)

