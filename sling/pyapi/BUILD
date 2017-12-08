package(default_visibility = ["//visibility:public"])

cc_library(
  name = "pyapi",
  srcs = [
    "pyapi.cc",
    "pyarray.cc",
    "pybase.cc",
    "pyframe.cc",
    "pystore.cc",
  ],
  hdrs = [
    "pyarray.h",
    "pybase.h",
    "pyframe.h",
    "pystore.h",
  ],
  deps = [
    "//sling/base",
    "//sling/file",
    "//sling/frame",
    "//sling/stream:file",
    "//sling/stream:unix-file",
  ],
  copts = [
    "-Wno-pmf-conversions",
  ]
)

cc_binary(
  name = "pysling.so",
  deps = [
    ":pyapi",
    "//sling/base",
    "//sling/base:libinit",
    "//sling/file:posix",
  ],
  linkshared = 1,
  linkstatic = 1,
)

