package(default_visibility = ["//visibility:public"])

cc_library(
  name = "wiki",
  srcs = ["wiki.cc"],
  hdrs = ["wiki.h"],
  deps = [
    "//sling/base",
    "//sling/frame:object",
    "//sling/frame:store",
    "//sling/util:unicode",
  ],
)

cc_library(
  name = "wikidata-converter",
  srcs = ["wikidata-converter.cc"],
  hdrs = ["wikidata-converter.h"],
  deps = [
    ":wiki",
    "//sling/frame",
    "//sling/nlp/kb:calendar",
    "//sling/string:text",
    "//sling/string:numbers",
  ],
)

cc_library(
  name = "wikipedia-map",
  srcs = ["wikipedia-map.cc"],
  hdrs = ["wikipedia-map.h"],
  deps = [
    ":wiki",
    "//sling/frame",
    "//sling/string:text",
  ],
)

cc_library(
  name = "wiki-parser",
  srcs = ["wiki-parser.cc"],
  hdrs = ["wiki-parser.h"],
  deps = [
    "//sling/base",
    "//sling/string:ctype",
    "//sling/string:printf",
    "//sling/string:strcat",
    "//sling/string:text",
  ],
)

cc_library(
  name = "wiki-extractor",
  srcs = ["wiki-extractor.cc"],
  hdrs = ["wiki-extractor.h"],
  deps = [
    ":wiki-parser",
    "//sling/string:strcat",
  ],
)

cc_library(
  name = "wiki-annotator",
  srcs = ["wiki-annotator.cc"],
  hdrs = ["wiki-annotator.h"],
  deps = [
    ":wiki",
    ":wiki-extractor",
    "//sling/base:registry",
    "//sling/frame",
    "//sling/nlp/document",
    "//sling/string:numbers",
    "//sling/string:strcat",
    "//sling/string:text",
  ],
)

cc_library(
  name = "wiki-macros",
  srcs = ["wiki-macros.cc"],
  deps = [
    ":wiki",
    ":wiki-annotator",
    "//sling/nlp/document",
    "//sling/nlp/document:fingerprinter",
    "//sling/nlp/document:lex",
    "//sling/nlp/document:document-tokenizer",
    "//sling/nlp/kb:calendar",
    "//sling/string:numbers",
    "//sling/util:mutex",
  ],
  alwayslink = 1,
)

cc_library(
  name = "wikidata-importer",
  srcs = ["wikidata-importer.cc"],
  deps = [
  ":wiki",
  ":wikidata-converter",
    "//sling/frame",
    "//sling/stream:input",
    "//sling/stream:memory",
    "//sling/string:text",
    "//sling/string:numbers",
    "//sling/task",
    "//sling/task:frames",
    "//sling/task:reducer",
  ],
  alwayslink = 1,
)

cc_library(
  name = "wikipedia-importer",
  srcs = ["wikipedia-importer.cc"],
  deps = [
    ":wiki",
    "//sling/base",
    "//sling/file",
    "//sling/frame",
    "//sling/stream:file-input",
    "//sling/string:numbers",
    "//sling/string:printf",
    "//sling/task",
    "//sling/task:frames",
    "//sling/task:process",
    "//sling/web:xml-parser",
  ],
  alwayslink = 1,
)

cc_library(
  name = "wikipedia-documents",
  srcs = ["wikipedia-documents.cc"],
  deps = [
    ":wiki-annotator",
    ":wiki-extractor",
    ":wiki-macros",
    ":wiki-parser",
    ":wikipedia-map",
    "//sling/base",
    "//sling/frame",
    "//sling/nlp/document",
    "//sling/nlp/document:document-tokenizer",
    "//sling/task",
    "//sling/task:frames",
    "//sling/task:reducer",
    "//sling/string:strcat",
    "//sling/string:text",
  ],
  alwayslink = 1,
)

cc_library(
  name = "wikipedia-links",
  srcs = ["wikipedia-links.cc"],
  deps = [
    "//sling/base",
    "//sling/nlp/document",
    "//sling/nlp/document:document-tokenizer",
    "//sling/nlp/document:lex",
    "//sling/task",
    "//sling/task:accumulator",
    "//sling/task:documents",
    "//sling/task:reducer",
  ],
  alwayslink = 1,
)

cc_library(
  name = "aliases",
  srcs = ["aliases.cc"],
  deps = [
    ":wiki",
    "//sling/base",
    "//sling/file:textmap",
    "//sling/frame",
    "//sling/nlp/document:phrase-tokenizer",
    "//sling/task",
    "//sling/task:frames",
    "//sling/task:reducer",
    "//sling/util:unicode",
  ],
  alwayslink = 1,
)

cc_binary(
  name = "parse-wiki-text",
  srcs = ["parse-wiki-text.cc"],
  deps = [
    ":wiki",
    ":wiki-annotator",
    ":wiki-extractor",
    ":wiki-macros",
    ":wiki-parser",
    ":wikipedia-map",
    "//sling/base",
    "//sling/file",
    "//sling/file:posix",
    "//sling/frame",
    "//sling/nlp/document",
    "//sling/nlp/document:document-tokenizer",
    "//sling/string",
  ],
)

