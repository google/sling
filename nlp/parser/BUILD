package(default_visibility = ["//visibility:public"])

cc_library(
  name = "parser-action",
  srcs = ["parser-action.cc"],
  hdrs = ["parser-action.h"],
  deps = [
    "//base",
    "//frame:object",
    "//frame:store",
    "//string:strcat",
    "//util:fingerprint",
  ],
)

cc_library(
  name = "parser-state",
  srcs = ["parser-state.cc"],
  hdrs = ["parser-state.h"],
  deps = [
    ":parser-action",
    "//base",
    "//frame:object",
    "//frame:store",
    "//nlp/document",
    "//string:strcat",
  ],
)

cc_library(
  name = "action-table",
  srcs = ["action-table.cc"],
  hdrs = ["action-table.h"],
  deps = [
    ":parser-action",
    ":parser-state",
    "//base",
    "//file",
    "//frame:object",
    "//frame:serialization",
    "//frame:store",
    "//string:numbers",
    "//string:strcat",
    "//string:text",
    "//util:table-writer",
  ],
)

cc_library(
  name = "roles",
  srcs = ["roles.cc"],
  hdrs = ["roles.h"],
  deps = [
    ":action-table",
    ":parser-state",
    "//frame:object",
  ],
)

cc_library(
  name = "parser",
  srcs = ["parser.cc"],
  hdrs = ["parser.h"],
  deps = [
    ":action-table",
    ":parser-state",
    ":roles",
    "//base",
    "//frame:serialization",
    "//frame:store",
    "//myelin:compute",
    "//myelin:flow",
    "//myelin/kernel:dragnn",
    "//myelin/kernel:tensorflow",
    "//nlp/document",
    "//nlp/document:features",
    "//nlp/document:lexicon",
  ],
)

