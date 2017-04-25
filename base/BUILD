package(default_visibility = ["//visibility:public"])

cc_library(
  name = "base",
  srcs = [
    "init.cc",
    "status.cc",
    "strtoint.cc",
  ],
  hdrs = [
    "bitcast.h",
    "flags.h",
    "init.h",
    "logging.h",
    "macros.h",
    "port.h",
    "slice.h",
    "status.h",
    "strtoint.h",
    "types.h",
  ],
  deps = [
    "//external:gflags",
    "//external:glog",
  ],
  copts = [
    "-Wno-sign-compare",
    "-Wno-deprecated",
  ],
  linkopts = [
    "-lpthread",
  ],
)

cc_library(
  name = "registry",
  srcs = ["registry.cc"],
  hdrs = ["registry.h"],
  deps = [
    ":base",
  ],
  copts = [
    "-Wno-sign-compare",
  ],
)

cc_library(
  name = "clock",
  srcs = ["clock.cc"],
  hdrs = ["clock.h"],
  deps = [
    ":base",
  ],
)

