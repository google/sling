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
    ":snapshot",
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
  name = "scanner",
  srcs = ["scanner.cc"],
  hdrs = ["scanner.h"],
  deps = [
    "//sling/base",
    "//sling/stream:input",
    "//sling/string:ctype",
    "//sling/string:strcat",
  ],
)

cc_library(
  name = "tokenizer",
  srcs = ["tokenizer.cc"],
  hdrs = ["tokenizer.h"],
  deps = [
    ":scanner",
    "//sling/base",
    "//sling/stream:input",
    "//sling/string:ctype",
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
  name = "xml",
  srcs = ["xml.cc"],
  hdrs = ["xml.h"],
  deps = [
    ":object",
    ":store",
    "//sling/base",
    "//sling/stream:input",
    "//sling/web:xml-parser",
  ],
)

cc_library(
  name = "turtle",
  srcs = ["turtle.cc"],
  hdrs = ["turtle.h"],
  deps = [
    ":object",
    ":scanner",
    ":store",
    "//sling/base",
    "//sling/stream:input",
    "//sling/string:ctype",
    "//sling/string:numbers",
  ],
)

cc_library(
  name = "snapshot",
  srcs = ["snapshot.cc"],
  hdrs = ["snapshot.h"],
  deps = [
    ":store",
    "//sling/base",
    "//sling/file",
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
    ":snapshot",
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

