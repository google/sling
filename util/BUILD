package(default_visibility = ["//visibility:public"])

cc_library(
  name = "varint",
  hdrs = ["varint.h"],
  srcs = ["varint.cc"],
  deps = [
    "//base",
  ],
)

cc_library(
  name = "hash",
  srcs = ["hash.cc"],
  hdrs = ["hash.h"],
  deps = [
    "//base",
  ],
)

cc_library(
  name = "city",
  srcs = ["city.cc"],
  hdrs = ["city.h"],
)

cc_library(
  name = "fingerprint",
  srcs = ["fingerprint.cc"],
  hdrs = ["fingerprint.h"],
  deps = [
    "//base",
  ],
)

cc_library(
  name = "unicode",
  hdrs = [
    "unicode.h",
  ],
  srcs = [
    "unicode.cc",
    "unicodetab.cc",
  ],
  deps = [
    "//base",
  ],
)

cc_library(
  name = "table-writer",
  srcs = ["table-writer.cc"],
  hdrs = ["table-writer.h"],
  deps = [
    "//base",
    "//file",
    "//string:numbers",
    "//string:printf",
    "//string:strcat",
  ],
)

cc_library(
  name = "zip-iterator",
  srcs = ["zip-iterator.cc"],
  hdrs = ["zip-iterator.h"],
  deps = [
    "//base",
    "//file:file",
    "//file:posix",
    "//third_party/zlib:miniunzip",
  ],
)

cc_binary(
  name = "zip-iterator-test",
  srcs = ["zip-iterator-test.cc"],
  deps = [
    ":zip-iterator",
    "//base",
    "//file:file",
    "//file:posix",
    "//stream:zipfile",
  ],
)
