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
    "//sling/string:strcat",
    "//sling/string:text",
  ],
)

cc_library(
  name = "wikidata-importer",
  srcs = ["wikidata-importer.cc"],
  deps = [
    ":wiki",
    "//sling/frame",
    "//sling/nlp/kb:calendar",
    "//sling/stream:input",
    "//sling/stream:output",
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
    ":wiki-parser",
    ":wikipedia-map",
    "//sling/base",
    "//sling/frame",
    "//sling/nlp/document:text-tokenizer",
    "//sling/nlp/document:tokens",
    "//sling/task",
    "//sling/task:frames",
    "//sling/task:reducer",
    "//sling/string:strcat",
    "//sling/string:text",
  ],
  alwayslink = 1,
)

cc_binary(
  name = "parse-wiki-text",
  srcs = ["parse-wiki-text.cc"],
  deps = [
    ":wiki-parser",
    "//sling/base",
    "//sling/file",
    "//sling/file:posix",
    "//sling/string",
  ],
)

