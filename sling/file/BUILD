package(default_visibility = ["//visibility:public"])

# File system interface.

cc_library(
  name = "file",
  srcs = ["file.cc"],
  hdrs = ["file.h"],
  deps = [
    "//sling/base",
    "//sling/base:registry",
  ],
  copts = [
    "-pthread",
  ],
)

cc_library(
  name = "posix",
  srcs = ["posix.cc"],
  hdrs = ["posix.h"],
  deps = [
    ":file",
    "//sling/base",
  ],
  alwayslink = 1,
)

cc_library(
  name = "embed",
  srcs = ["embed.cc"],
  hdrs = ["embed.h"],
  deps = [
    ":file",
    "//sling/base",
  ],
  alwayslink = 1,
)

# File utility libraries.

cc_library(
  name = "recordio",
  srcs = ["recordio.cc"],
  hdrs = ["recordio.h"],
  deps = [
    ":file",
    "//sling/base",
    "//sling/util:fingerprint",
    "//sling/util:snappy",
    "//sling/util:varint",
  ],
)

cc_library(
  name = "textmap",
  srcs = ["textmap.cc"],
  hdrs = ["textmap.h"],
  deps = [
    ":file",
    "//sling/base",
    "//sling/string:text",
  ],
)

cc_library(
  name = "repository",
  srcs = ["repository.cc"],
  hdrs = ["repository.h"],
  deps = [
    ":file",
    "//sling/base",
  ],
)

