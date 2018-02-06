package(default_visibility = ["//visibility:public"])

cc_library(
  name = "pyapi",
  srcs = [
    "pyapi.cc",
    "pyarray.cc",
    "pybase.cc",
    "pyframe.cc",
    "pyparser.cc",
    "pyrecordio.cc",
    "pystore.cc",
  ],
  hdrs = [
    "pyarray.h",
    "pybase.h",
    "pyframe.h",
    "pyparser.h",
    "pyrecordio.h",
    "pystore.h",
  ],
  deps = [
    "//sling/base",
    "//sling/file",
    "//sling/file:recordio",
    "//sling/frame",
    "//sling/nlp/document",
    "//sling/nlp/document:document-tokenizer",
    "//sling/nlp/parser",
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
    "//sling/file:posix",
  ],
  linkshared = 1,
  linkstatic = 1,
)

