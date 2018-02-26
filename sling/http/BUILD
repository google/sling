package(default_visibility = ["//visibility:public"])

cc_library(
  name = "http-server",
  srcs = ["http-server.cc"],
  hdrs = ["http-server.h"],
  deps = [
    "//sling/base",
    "//sling/file",
    "//sling/string:ctype",
    "//sling/string:numbers",
    "//sling/util:mutex",
    "//sling/util:thread",
  ],
)

cc_library(
  name = "static-content",
  srcs = ["static-content.cc"],
  hdrs = ["static-content.h"],
  deps = [
    ":http-server",
    "//sling/base",
    "//sling/file",
  ],
)

