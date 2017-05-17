package(default_visibility = ["//visibility:public"])

cc_library(
    name = "gold-transition-generator",
    srcs = ["gold-transition-generator.cc"],
    hdrs = ["gold-transition-generator.h"],
    deps = [
        "//base",
        "//nlp/document",
        "//nlp/parser:parser-action",
        "//frame:object",
        "//frame:store",
        "//string:strcat",
    ],
)

cc_library(
    name = "action-table-generator",
    srcs = ["action-table-generator.cc"],
    hdrs = ["action-table-generator.h"],
    deps = [
        ":gold-transition-generator",
        "//base",
        "//nlp/parser:action-table",
        "//nlp/parser:parser-state",
        "//file",
        "//frame:object",
        "//frame:serialization",
        "//frame:store",
        "//nlp/document",
        "//string:strcat",
    ],
)

cc_library(
    name = "frame-evaluation",
    srcs = ["frame-evaluation.cc"],
    hdrs = [],
    deps = [
        "//base",
        "//nlp/document",
        "//frame:object",
        "//frame:store",
        "//frame:serialization",
        "//util/hash",
    ],
    alwayslink = 1,
)

cc_binary(
  name = "action-table-generator-main",
  srcs = ["action-table-generator-main.cc"],
  deps = [
    ":action-table-generator",
    "//base",
    "//file:file",
    "//file:posix",
    "//frame:object",
    "//frame:serialization",
    "//frame:store",
    "//string:strcat"
  ],
)
