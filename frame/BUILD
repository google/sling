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
    "//base",
    "//base:clock",
    "//string:strcat",
    "//string:text",
    "//util:city",
  ],
  copts = ["-Wno-maybe-uninitialized"],
)

cc_library(
  name = "object",
  srcs = ["object.cc"],
  hdrs = ["object.h"],
  deps = [
    ":store",
    "//base",
    "//string:text",
  ],
)

cc_library(
  name = "tokenizer",
  srcs = ["tokenizer.cc"],
  hdrs = ["tokenizer.h"],
  deps = [
    "//base",
    "//stream:input",
    "//string:ctype",
    "//string:numbers",
    "//string:strcat",
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
    "//base",
    "//stream:input",
    "//string:numbers",
  ],
)

cc_library(
  name = "printer",
  srcs = ["printer.cc"],
  hdrs = ["printer.h"],
  deps = [
    ":object",
    ":store",
    "//base",
    "//stream:output",
    "//string:ctype",
    "//string:numbers",
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
    "//base",
    "//stream:output",
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
    "//base",
    "//stream:input",
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
    "//base",
    "//file",
    "//stream",
    "//stream:file",
    "//stream:memory",
    "//string:text",
  ],
)

