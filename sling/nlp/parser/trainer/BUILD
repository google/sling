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

