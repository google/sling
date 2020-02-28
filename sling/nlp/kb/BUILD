package(default_visibility = ["//visibility:public"])

load("//tools:embed.bzl", "embed_data")

cc_library(
  name = "calendar",
  srcs = ["calendar.cc"],
  hdrs = ["calendar.h"],
  deps = [
    "//sling/frame:object",
    "//sling/frame:store",
    "//sling/string:text",
    "//sling/string:strcat",
    "//sling/util:unicode",
  ],
)

cc_library(
  name = "facts",
  srcs = ["facts.cc"],
  hdrs = ["facts.h"],
  deps = [
    ":calendar",
    "//sling/frame:object",
    "//sling/frame:store",
  ],
)

cc_library(
  name = "name-table-builder",
  srcs = ["name-table-builder.cc"],
  deps = [
    "//sling/base",
    "//sling/file:repository",
    "//sling/frame:object",
    "//sling/task",
    "//sling/task:frames",
    "//sling/util:mutex",
    "//sling/util:unicode",
  ],
  alwayslink = 1,
)

cc_library(
  name = "name-table",
  srcs = ["name-table.cc"],
  hdrs = ["name-table.h"],
  deps = [
    "//sling/base",
    "//sling/file:repository",
    "//sling/string:text",
    "//sling/util:unicode",
  ],
)

cc_library(
  name = "phrase-table-builder",
  srcs = ["phrase-table-builder.cc"],
  deps = [
    ":facts",
    "//sling/base",
    "//sling/file:repository",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/nlp/document:phrase-tokenizer",
    "//sling/nlp/wiki",
    "//sling/task",
    "//sling/task:frames",
    "//sling/util:mutex",
  ],
  alwayslink = 1,
)

cc_library(
  name = "phrase-table",
  srcs = ["phrase-table.cc"],
  hdrs = ["phrase-table.h"],
  deps = [
    "//sling/base",
    "//sling/file:repository",
    "//sling/frame:store",
    "//sling/frame:object",
    "//sling/string:text",
    "//sling/util:asset",
  ],
)

cc_library(
  name = "fact-lexicon",
  srcs = ["fact-lexicon.cc"],
  deps = [
    ":facts",
    "//sling/base",
    "//sling/file:textmap",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/task:frames",
    "//sling/task:process",
    "//sling/util:bloom",
    "//sling/util:sortmap",
  ],
  alwayslink = 1,
)

cc_library(
  name = "resolver",
  srcs = ["resolver.cc"],
  hdrs = ["resolver.h"],
  deps = [
    ":phrase-table",
    "//sling/base",
    "//sling/frame:object",
    "//sling/frame:store",
    "//sling/util:top",
    "//sling/util:unicode",
  ],
)

embed_data(
  name = "app",
  srcs = [
    "app/appicon.ico",
    "app/index.html",
    "app/kb.css",
    "app/kb.js",
  ],
)

cc_library(
  name = "knowledge-service",
  srcs = ["knowledge-service.cc"],
  hdrs = ["knowledge-service.h"],
  deps = [
    ":app",
    ":calendar",
    ":name-table",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/http:http-server",
    "//sling/http:static-content",
    "//sling/http:web-service",
  ],
)

cc_binary(
  name = "knowledge-server",
  srcs = ["knowledge-server.cc"],
  deps = [
    ":knowledge-service",
    "//sling/base",
    "//sling/file",
    "//sling/file:embed",
    "//sling/file:posix",
    "//sling/http:http-server",
    "//sling/http:web-service",
    "//sling/string:text",
    "//sling/string:strcat",
  ],
)

