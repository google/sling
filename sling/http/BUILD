package(default_visibility = ["//visibility:public"])

cc_library(
  name = "http-utils",
  srcs = ["http-utils.cc"],
  hdrs = ["http-utils.h"],
  deps = [
    "//sling/base",
    "//sling/string:ctype",
  ],
)

cc_library(
  name = "http-server",
  srcs = ["http-server.cc"],
  hdrs = ["http-server.h"],
  deps = [
    ":http-utils",
    "//sling/base",
    "//sling/file",
    "//sling/string:numbers",
    "//sling/util:mutex",
    "//sling/util:thread",
  ],
)

cc_library(
  name = "http-stream",
  srcs = ["http-stream.cc"],
  hdrs = ["http-stream.h"],
  deps = [
    ":http-server",
    "//sling/stream",
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

cc_library(
  name = "web-service",
  srcs = ["web-service.cc"],
  hdrs = ["web-service.h"],
  deps = [
    ":http-server",
    ":http-stream",
    "//sling/base",
    "//sling/frame:decoder",
    "//sling/frame:encoder",
    "//sling/frame:json",
    "//sling/frame:printer",
    "//sling/frame:object",
    "//sling/frame:reader",
    "//sling/frame:store",
    "//sling/string:text",
  ],
)

