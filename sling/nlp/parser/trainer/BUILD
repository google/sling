package(default_visibility = ["//visibility:public"])

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
    "//sling/nlp/document:document-corpus",
    "//sling/string:strcat",
    "//sling/string:printf",
  ],
)

cc_library(
  name = "transition-generator",
  hdrs = ["transition-generator.h"],
  srcs = ["transition-generator.cc"],
  deps = [
    "//sling/base",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/nlp/document:document",
    "//sling/nlp/parser:parser-action",
  ],
)

cc_binary(
  name = "generate-transitions",
  srcs = ["generate-transitions.cc"],
  deps = [
    ":transition-generator",
    "//sling/base",
    "//sling/file:posix",
    "//sling/frame:store",
    "//sling/nlp/document:document",
    "//sling/nlp/document:document-corpus",
    "//sling/nlp/parser:parser-action",
  ],
)

