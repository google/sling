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
    ":calendar",
    ":app",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/http:http-server",
    "//sling/http:static-content",
    "//sling/http:web-service",
    "//sling/nlp/wiki:name-table",
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
