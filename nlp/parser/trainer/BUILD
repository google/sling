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

cc_binary(
  name = "generate-action-table",
  srcs = ["generate-action-table.cc"],
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
