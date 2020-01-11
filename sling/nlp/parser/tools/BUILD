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
    "//sling/nlp/document:document-corpus",
    "//sling/nlp/document:document-tokenizer",
    "//sling/nlp/document:lex",
    "//sling/nlp/parser",
    "//sling/nlp/parser:multiclass-delegate",
    "//sling/nlp/parser:frame-evaluation",
    "//sling/string:printf",
  ],
)

