package(default_visibility = ["//visibility:public"])

cc_library(
  name = "xml-parser",
  srcs = ["xml-parser.cc"],
  hdrs = ["xml-parser.h"],
  deps = [
    ":entity-ref",
    "//sling/base",
    "//sling/stream:input",
    "//sling/string:ctype",
    "//sling/util:unicode",
  ],
)

cc_library(
  name = "entity-ref",
  srcs = ["entity-ref.cc"],
  hdrs = ["entity-ref.h"],
  deps = [
    "//sling/base",
  ],
)

