package(default_visibility = ["//visibility:public"])

load("//tools:embed.bzl", "embed_data")

cc_library(
  name = "token-properties",
  hdrs = ["token-properties.h"],
)

cc_library(
  name = "document",
  srcs = ["document.cc"],
  hdrs = ["document.h"],
  deps = [
    ":fingerprinter",
    ":token-properties",
    "//sling/base",
    "//sling/frame:object",
    "//sling/frame:store",
    "//sling/string:text",
    "//sling/util:unicode",
  ],
)

cc_library(
  name = "annotator",
  srcs = ["annotator.cc"],
  hdrs = ["annotator.h"],
  deps = [
    ":document",
    "//sling/base",
    "//sling/file",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/task:task",
  ],
)

cc_library(
  name = "text-tokenizer",
  srcs = ["text-tokenizer.cc"],
  hdrs = ["text-tokenizer.h"],
  deps = [
    ":token-properties",
    "//sling/base",
    "//sling/string:ctype",
    "//sling/string:text",
    "//sling/util:unicode",
    "//sling/web:entity-ref",
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
    "//sling/util:unicode",
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
  name = "document-corpus",
  srcs = ["document-corpus.cc"],
  hdrs = ["document-corpus.h"],
  deps = [
    ":document",
    "//sling/base",
    "//sling/file:recordio",
    "//sling/frame:serialization",
    "//sling/frame:store",
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
    "//sling/util:unicode",
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
    "//sling/util:unicode",
  ],
)

cc_library(
  name = "vocabulary-builder",
  srcs = ["vocabulary-builder.cc"],
  deps = [
    "//sling/base",
    "//sling/file:textmap",
    "//sling/nlp/document",
    "//sling/task:accumulator",
    "//sling/task:documents",
    "//sling/util:unicode",
  ],
  alwayslink = 1,
)

cc_library(
  name = "document-service",
  srcs = ["document-service.cc"],
  hdrs = ["document-service.h"],
  deps = [
    ":document",
    ":document-tokenizer",
    ":lex",
    "//sling/base",
    "//sling/frame",
    "//sling/http:web-service",
  ],
)

embed_data(
  name = "corpus-app",
  srcs = [
    "app/corpus.html",
    "app/corpus.js",
    "app/corpus.css",
  ],
)

cc_binary(
  name = "corpus-browser",
  srcs = ["corpus-browser.cc"],
  deps = [
    ":annotator",
    ":annotator-components",
    ":corpus-app",
    ":document",
    ":document-service",
    "//app",
    "//sling/base",
    "//sling/file",
    "//sling/file:embed",
    "//sling/file:posix",
    "//sling/file:recordio",
    "//sling/frame",
    "//sling/http:http-server",
    "//sling/http:static-content",
    "//sling/http:web-service",
    "//sling/nlp/kb:knowledge-service",
    "//sling/util:mutex",
  ],
)

cc_library(
  name = "annotator-components",
  deps = [
    "//sling/nlp/parser:parser-annotator",
    "//sling/nlp/silver:annotators",
  ],
)

embed_data(
  name = "analyzer-app",
  srcs = [
    "app/analyzer.html",
    "app/analyzer.js",
    "app/analyzer.css",
  ],
)

cc_binary(
  name = "analyzer",
  srcs = ["analyzer.cc"],
  deps = [
    ":analyzer-app",
    ":annotator",
    ":annotator-components",
    ":document",
    ":document-service",
    "//app",
    "//sling/base",
    "//sling/file",
    "//sling/file:embed",
    "//sling/file:posix",
    "//sling/frame",
    "//sling/http:http-server",
    "//sling/http:static-content",
    "//sling/http:web-service",
    "//sling/nlp/kb:knowledge-service",
  ],
)

