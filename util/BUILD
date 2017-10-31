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
  name = "vocabulary",
  srcs = ["vocabulary.cc"],
  hdrs = ["vocabulary.h"],
  deps = [
    ":fingerprint",
    "//base",
  ],
)

cc_library(
  name = "embeddings",
  srcs = ["embeddings.cc"],
  hdrs = ["embeddings.h"],
  deps = [
    "//base",
    "//stream:file",
    "//stream:input",
    "//stream:output",
  ],
)

