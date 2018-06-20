package(default_visibility = ["//visibility:public"])

cc_library(
  name = "token-breaks",
  hdrs = ["token-breaks.h"],
)

cc_library(
  name = "document",
  srcs = ["document.cc"],
  hdrs = ["document.h"],
  deps = [
    ":fingerprinter",
    ":token-breaks",
    "//sling/base",
    "//sling/frame:object",
    "//sling/frame:store",
    "//sling/string:text",
  ],
)

cc_library(
  name = "text-tokenizer",
  srcs = ["text-tokenizer.cc"],
  hdrs = ["text-tokenizer.h"],
  deps = [
    ":token-breaks",
    "//sling/base",
    "//sling/string:ctype",
    "//sling/string:text",
    "//sling/util:unicode",
    "//sling/web:entity-ref",
  ],
)

cc_library(
  name = "tokens",
  srcs = ["tokens.cc"],
  hdrs = ["tokens.h"],
  deps = [
    ":text-tokenizer",
    "//sling/base",
    "//sling/frame:object",
    "//sling/frame:store",
  ],
)

cc_library(
  name = "fingerprinter",
  srcs = ["fingerprinter.cc"],
  hdrs = ["fingerprinter.h"],
  deps = [
    "//sling/base",
    "//sling/string:text",
    "//sling/util:fingerprint",
    "//sling/util:unicode",
  ],
)

cc_library(
  name = "phrase-tokenizer",
  srcs = ["phrase-tokenizer.cc"],
  hdrs = ["phrase-tokenizer.h"],
  deps = [
    ":fingerprinter",
    ":text-tokenizer",
    "//sling/base",
    "//sling/string:text",
  ],
)

cc_library(
  name = "document-tokenizer",
  srcs = ["document-tokenizer.cc"],
  hdrs = ["document-tokenizer.h"],
  deps = [
    ":document",
    ":text-tokenizer",
    "//sling/base",
    "//sling/frame:object",
    "//sling/frame:store",
    "//sling/string:text",
  ],
)

cc_library(
  name = "lex",
  srcs = ["lex.cc"],
  hdrs = ["lex.h"],
  deps = [
    ":document",
    ":document-tokenizer",
    "//sling/base",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/string:text",
    "//sling/string:ctype",
  ],
)

cc_library(
  name = "document-source",
  srcs = ["document-source.cc"],
  hdrs = ["document-source.h"],
  deps = [
    ":document",
    "//sling/base",
    "//sling/file:recordio",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/stream:input",
    "//sling/stream:stream",
    "//sling/stream:zipfile",
    "//sling/string",
  ],
)

cc_library(
  name = "affix",
  srcs = ["affix.cc"],
  hdrs = ["affix.h"],
  deps = [
    "//sling/base",
    "//sling/stream:input",
    "//sling/stream:output",
    "//sling/string:text",
    "//sling/util:fingerprint",
    "//sling/util:unicode",
  ],
)

cc_library(
  name = "lexicon",
  srcs = ["lexicon.cc"],
  hdrs = ["lexicon.h"],
  deps = [
    ":affix",
    "//sling/base",
    "//sling/stream:memory",
    "//sling/util:vocabulary",
  ],
)

cc_library(
  name = "features",
  srcs = ["features.cc"],
  hdrs = ["features.h"],
  deps = [
    ":document",
    ":lexicon",
    "//sling/util:unicode",
  ],
)

cc_library(
  name = "lexical-encoder",
  srcs = ["lexical-encoder.cc"],
  hdrs = ["lexical-encoder.h"],
  deps = [
    ":document",
    ":features",
    ":lexicon",
    "//sling/myelin:builder",
    "//sling/myelin:compute",
    "//sling/myelin:flow",
    "//sling/myelin:gradient",
    "//sling/myelin:rnn",
    "//sling/util:embeddings",
  ],
)

