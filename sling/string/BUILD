package(default_visibility = ["//visibility:public"])

cc_library(
  name = "string",
  deps = [
    ":ctype",
    ":numbers",
    ":printf",
    ":strcat",
    ":text",
  ],
)

cc_library(
  name = "ctype",
  hdrs = ["ctype.h"],
  srcs = ["ctype.cc"],
  copts = [
    "-Wno-narrowing",
  ]
)

cc_library(
  name = "numbers",
  hdrs = ["numbers.h"],
  srcs = ["numbers.cc"],
  deps = [
    ":ctype",
    "//sling/base",
  ],
)

cc_library(
  name = "printf",
  hdrs = ["printf.h"],
  srcs = ["printf.cc"],
  deps = [
    "//sling/base",
  ],
)

cc_library(
  name = "strcat",
  hdrs = ["strcat.h"],
  srcs = ["strcat.cc"],
  deps = [
    ":ctype",
    ":numbers",
    ":text",
    "//sling/base",
  ],
)

cc_library(
  name = "text",
  hdrs = ["text.h"],
  srcs = ["text.cc"],
  deps = [
    "//sling/base",
    "//sling/util:city",
  ],
  copts = [
    "-Wno-deprecated",
    "-Wno-sign-compare",
  ],
)

