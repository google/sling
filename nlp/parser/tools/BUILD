package(default_visibility = ["//visibility:public"])

cc_binary(
  name = "parse",
  srcs = ["parse.cc"],
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

cc_binary(
  name = "evaluate-frames",
  srcs = ["evaluate-frames.cc"],
  deps = [
    "//base",
    "//file:posix",
    "//nlp/parser/trainer:frame-evaluation",
    "//third_party/zlib:miniunzip",
    "//third_party/zlib:zlib",
  ],
)

