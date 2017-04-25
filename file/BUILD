package(default_visibility = ["//visibility:public"])

# File system interface.

cc_library(
  name = "file",
  srcs = ["file.cc"],
  hdrs = ["file.h"],
  deps = [
    "//base",
    "//base:registry",
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
    "//base",
  ],
  alwayslink = 1,
)

