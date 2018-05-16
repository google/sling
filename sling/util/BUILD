package(default_visibility = ["//visibility:public"])

cc_library(
  name = "varint",
  hdrs = ["varint.h"],
  srcs = ["varint.cc"],
  deps = [
    "//sling/base",
  ],
)

cc_library(
  name = "city",
  srcs = ["city.cc"],
  hdrs = ["city.h"],
  deps = [
    "//sling/base",
  ],
)

cc_library(
  name = "fingerprint",
  srcs = ["fingerprint.cc"],
  hdrs = ["fingerprint.h"],
  deps = [
    "//sling/base",
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
    "//sling/base",
  ],
)

cc_library(
  name = "random",
  hdrs = ["random.h"],
)

cc_library(
  name = "thread",
  srcs = ["thread.cc"],
  hdrs = [
    "thread.h",
  ],
  deps = [
    "//sling/base",
  ],
  copts = [
    "-pthread",
  ],
)

cc_library(
  name = "threadpool",
  srcs = ["threadpool.cc"],
  hdrs = ["threadpool.h"],
  deps = [
    ":thread",
    "//sling/base",
  ],
  copts = [
    "-pthread",
  ],
)

cc_library(
  name = "mutex",
  hdrs = ["mutex.h"],
  copts = [
    "-pthread",
  ],
)

cc_library(
  name = "table-writer",
  srcs = ["table-writer.cc"],
  hdrs = ["table-writer.h"],
  deps = [
    "//sling/base",
    "//sling/file",
    "//sling/string:numbers",
    "//sling/string:printf",
    "//sling/string:strcat",
  ],
)

cc_library(
  name = "vocabulary",
  srcs = ["vocabulary.cc"],
  hdrs = ["vocabulary.h"],
  deps = [
    ":fingerprint",
    "//sling/base",
    "//sling/string:text",
  ],
)

cc_library(
  name = "embeddings",
  srcs = ["embeddings.cc"],
  hdrs = ["embeddings.h"],
  deps = [
    "//sling/base",
    "//sling/stream:file",
    "//sling/stream:input",
    "//sling/stream:output",
  ],
)

cc_library(
  name = "elf-writer",
  srcs = ["elf-writer.cc"],
  hdrs = ["elf-writer.h"],
  copts = [
    "-Wno-sign-compare",
  ],
)
