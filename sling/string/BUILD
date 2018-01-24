package(default_visibility = ["//visibility:public"])

cc_library(
  name = "string",
  deps = [
    ":ctype",
    ":memutil",
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
  name = "memutil",
  hdrs = ["memutil.h"],
  srcs = ["memutil.cc"],
  deps = [
    ":ctype",
    "//sling/base",
  ],
  copts = [
    "-Wno-sign-compare",
    "-Wno-deprecated",
  ],
)

cc_library(
  name = "numbers",
  hdrs = ["numbers.h"],
  srcs = ["numbers.cc"],
  deps = [
    ":ctype",
    ":printf",
    "//sling/base",
  ],
  copts = [
    "-Wno-narrowing",
    "-Wno-sign-compare",
    "-Wno-deprecated",
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
  copts = [
    "-Wno-sign-compare",
    "-Wno-deprecated",
  ],
)

cc_library(
  name = "text",
  hdrs = ["text.h"],
  srcs = ["text.cc"],
  deps = [
    ":memutil",
    "//sling/base",
    "//sling/util:city",
  ],
  copts = [
    "-Wno-deprecated",
    "-Wno-sign-compare",
  ],
)

