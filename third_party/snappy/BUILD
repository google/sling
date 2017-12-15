# Snappy compression library, see https://github.com/google/snappy

package(default_visibility = ["//visibility:public"])

licenses(["notice"]) # BSD

exports_files(["LICENSE"])

cc_library(
  name = "bits",
  hdrs = ["bits.h"],
  deps = [
    "//sling/base",
  ],
)

cc_library(
  name = "endian",
  hdrs = ["endian.h"],
  deps = [
    "//sling/base",
  ],
)

cc_library(
  name = "snappy",
  srcs = [
    "snappy.cc",
    "snappy-sinksource.cc",
  ],
  hdrs = [
    "snappy.h",
    "snappy-sinksource.h",
  ],
  deps = [
    ":bits",
    ":endian",
    "//sling/base",
    "//sling/util:varint",
  ],
  copts = [
    "-Wno-sign-compare",
  ],
)

