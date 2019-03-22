package(default_visibility = ["//visibility:public"])

cc_library(
  name = "idf",
  srcs = ["idf.cc"],
  hdrs = ["idf.h"],
  deps = [
    "//sling/base",
    "//sling/file:repository",
    "//sling/nlp/document",
    "//sling/nlp/document:fingerprinter",
    "//sling/task:accumulator",
    "//sling/task:documents",
    "//sling/util:unicode",
  ],
  alwayslink = 1,
)

cc_library(
  name = "chart",
  srcs = ["chart.cc"],
  hdrs = ["chart.h"],
  deps = [
    "//sling/frame:store",
    "//sling/nlp/document",
    "//sling/nlp/kb:phrase-table",
  ],
)

cc_library(
  name = "resolver",
  srcs = ["resolver.cc"],
  hdrs = ["resolver.h"],
  deps = [
    "//sling/base",
    "//sling/frame:object",
    "//sling/frame:store",
    "//sling/nlp/kb:phrase-table",
    "//sling/util:top",
    "//sling/util:unicode",
  ],
)

cc_library(
  name = "annotators",
  srcs = ["annotators.cc"],
  hdrs = ["annotators.h"],
  deps = [
    ":chart",
    ":idf",
    ":resolver",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/nlp/document",
    "//sling/nlp/kb:calendar",
    "//sling/nlp/kb:facts",
    "//sling/nlp/kb:phrase-table",
  ],
)

cc_binary(
  name = "parse-chart",
  srcs = ["parse-chart.cc"],
  deps = [
    ":annotators",
    ":chart",
    ":idf",
    "//sling/base",
    "//sling/file:posix",
    "//sling/file:recordio",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/nlp/document",
    "//sling/nlp/document:document-tokenizer",
    "//sling/nlp/document:lex",
  ],
)

