package(default_visibility = ["//visibility:public"])

cc_library(
  name = "frame",
  deps = [
    ":decoder",
    ":encoder",
    ":object",
    ":printer",
    ":reader",
    ":serialization",
    ":store",
  ],
)

cc_library(
  name = "store",
  srcs = ["store.cc"],
  hdrs = ["store.h"],
  deps = [
    "//sling/base",
    "//sling/base:clock",
    "//sling/string:strcat",
    "//sling/string:text",
    "//sling/util:city",
  ],
)

cc_library(
  name = "object",
  srcs = ["object.cc"],
  hdrs = ["object.h"],
  deps = [
    ":store",
    "//sling/base",
    "//sling/string:text",
  ],
)

cc_library(
  name = "tokenizer",
  srcs = ["tokenizer.cc"],
  hdrs = ["tokenizer.h"],
  deps = [
    "//sling/base",
    "//sling/stream:input",
    "//sling/string:ctype",
    "//sling/string:numbers",
    "//sling/string:strcat",
  ],
)

cc_library(
  name = "reader",
  srcs = ["reader.cc"],
  hdrs = ["reader.h"],
  deps = [
    ":object",
    ":store",
    ":tokenizer",
    "//sling/base",
    "//sling/stream:input",
    "//sling/string:numbers",
  ],
)

cc_library(
  name = "printer",
  srcs = ["printer.cc"],
  hdrs = ["printer.h"],
  deps = [
    ":object",
    ":store",
    "//sling/base",
    "//sling/stream:output",
    "//sling/string:ctype",
    "//sling/string:numbers",
  ],
)

cc_library(
  name = "wire",
  hdrs = ["wire.h"],
)

cc_library(
  name = "encoder",
  srcs = ["encoder.cc"],
  hdrs = ["encoder.h"],
  deps = [
    ":object",
    ":store",
    ":wire",
    "//sling/base",
    "//sling/stream:output",
  ],
)

cc_library(
  name = "decoder",
  srcs = ["decoder.cc"],
  hdrs = ["decoder.h"],
  deps = [
    ":object",
    ":store",
    ":wire",
    "//sling/base",
    "//sling/stream:input",
  ],
)

cc_library(
  name = "json",
  srcs = ["json.cc"],
  hdrs = ["json.h"],
  deps = [
    ":object",
    ":store",
    "//sling/base",
    "//sling/stream:output",
    "//sling/string:numbers",
  ],
)

cc_library(
  name = "serialization",
  srcs = ["serialization.cc"],
  hdrs = ["serialization.h"],
  deps = [
    ":decoder",
    ":encoder",
    ":object",
    ":printer",
    ":reader",
    ":store",
    ":wire",
    "//sling/base",
    "//sling/file",
    "//sling/stream",
    "//sling/stream:file",
    "//sling/stream:memory",
    "//sling/string:text",
  ],
)

