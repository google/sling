package(default_visibility = ["//visibility:public"])

cc_library(
  name = "stream",
  hdrs = ["stream.h"],
  deps = [
    "//sling/base",
  ],
)

cc_library(
  name = "file",
  srcs = ["file.cc"],
  hdrs = ["file.h"],
  deps = [
    ":stream",
    "//sling/base",
    "//sling/file",
  ],
)

cc_library(
  name = "memory",
  srcs = ["memory.cc"],
  hdrs = ["memory.h"],
  deps = [
    ":stream",
    "//sling/base",
  ],
)

cc_library(
  name = "input",
  srcs = ["input.cc"],
  hdrs = ["input.h"],
  deps = [
    ":stream",
    "//sling/base",
    "//sling/util:varint",
  ],
)

cc_library(
  name = "output",
  srcs = ["output.cc"],
  hdrs = ["output.h"],
  deps = [
    ":stream",
    "//sling/base",
    "//sling/string:text",
    "//sling/util:varint",
  ],
)

cc_library(
  name = "file-input",
  srcs = ["file-input.cc"],
  hdrs = ["file-input.h"],
  deps = [
    ":bzip2",
    ":file",
    ":gzip",
    ":input",
  ],
)

cc_library(
  name = "bounded",
  srcs = ["bounded.cc"],
  hdrs = ["bounded.h"],
  deps = [
    ":stream",
  ],
)

cc_library(
  name = "bzip2",
  srcs = ["bzip2.cc"],
  hdrs = ["bzip2.h"],
  deps = [
    ":stream",
    "//sling/base",
    "//third_party/bz2lib",
  ],
)

cc_library(
  name = "gzip",
  srcs = ["gzip.cc"],
  hdrs = ["gzip.h"],
  deps = [
    ":stream",
    "//sling/base",
    "//third_party/zlib",
  ],
)

cc_library(
  name = "zipfile",
  srcs = ["zipfile.cc"],
  hdrs = ["zipfile.h"],
  deps = [
    ":bounded",
    ":file",
    ":file-input",
    ":gzip",
    "//sling/base",
    "//sling/file",
  ],
)

cc_library(
  name = "unix-file",
  srcs = ["unix-file.cc"],
  hdrs = ["unix-file.h"],
  deps = [
    ":stream",
    "//sling/base",
  ],
)

