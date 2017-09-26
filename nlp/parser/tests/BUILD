package(default_visibility = ["//visibility:public"])

cc_binary(
  name = "parser-test",
  srcs = ["parser-test.cc"],
  deps = [
    "//base",
    "//base:clock",
    "//file:posix",
    "//frame:object",
    "//frame:serialization",
    "//nlp/document",
    "//nlp/document:document-source",
    "//nlp/document:document-tokenizer",
    "//nlp/parser",
    "//nlp/parser/trainer:frame-evaluation",
    "//string:printf",
  ],
)

