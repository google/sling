package(default_visibility = ["//visibility:public"])

cc_library(
  name = "parser-action",
  srcs = ["parser-action.cc"],
  hdrs = ["parser-action.h"],
  deps = [
    "//sling/base",
    "//sling/frame:object",
    "//sling/frame:store",
    "//sling/string:strcat",
    "//sling/util:fingerprint",
  ],
)

cc_library(
  name = "parser-state",
  srcs = ["parser-state.cc"],
  hdrs = ["parser-state.h"],
  deps = [
    ":parser-action",
    "//sling/base",
    "//sling/frame:object",
    "//sling/frame:store",
    "//sling/nlp/document",
    "//sling/string:strcat",
  ],
)

cc_library(
  name = "action-table",
  srcs = ["action-table.cc"],
  hdrs = ["action-table.h"],
  deps = [
    ":parser-action",
    ":parser-state",
    "//sling/base",
    "//sling/file",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/string:numbers",
    "//sling/string:strcat",
    "//sling/string:text",
    "//sling/util:table-writer",
  ],
)

cc_library(
  name = "roles",
  srcs = ["roles.cc"],
  hdrs = ["roles.h"],
  deps = [
    ":action-table",
    ":parser-state",
    "//sling/frame:object",
  ],
)

cc_library(
  name = "cascade",
  srcs = ["cascade.cc"],
  hdrs = ["cascade.h"],
  deps = [
    ":action-table",
    ":parser-state",
    "//sling/base",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/myelin:compute",
    "//sling/myelin:flow",
  ],
)

cc_library(
  name = "parser",
  srcs = ["parser.cc"],
  hdrs = ["parser.h"],
  deps = [
    ":action-table",
    ":cascade",
    ":parser-state",
    ":roles",
    "//sling/base",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/myelin:compute",
    "//sling/myelin:flow",
    "//sling/myelin/kernel:cuda",
    "//sling/myelin/kernel:dragnn",
    "//sling/myelin/kernel:tensorflow",
    "//sling/nlp/document",
    "//sling/nlp/document:lexical-encoder",
    "//sling/string:strcat",
  ],
)
