package(default_visibility = ["//visibility:public"])

cc_binary(
  name = "parse",
  srcs = ["parse.cc"],
  deps = [
    "//sling/base",
    "//sling/base:clock",
    "//sling/file:posix",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/myelin:profile",
    "//sling/nlp/document",
    "//sling/nlp/document:document-source",
    "//sling/nlp/document:document-tokenizer",
    "//sling/nlp/parser",
    "//sling/nlp/parser/trainer:frame-evaluation",
    "//sling/string:printf",
  ],
)

cc_binary(
  name = "evaluate-frames",
  srcs = ["evaluate-frames.cc"],
  deps = [
    "//sling/base",
    "//sling/file:posix",
    "//sling/nlp/parser/trainer:frame-evaluation",
    "//third_party/zlib",
  ],
)

cc_binary(
  name = "build-store",
  srcs = ["build-store.cc"],
  deps = [
    "//sling/base",
    "//sling/file:posix",
    "//sling/frame:object",
    "//sling/frame:serialization",
  ],
)

