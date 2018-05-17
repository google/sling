package(default_visibility = ["//visibility:public"])

cc_library(
  name = "index",
  srcs = ["index.cc"],
  hdrs = ["index.h"],
  deps = [
    "//sling/base",
    "//sling/myelin:compute",
    "//sling/myelin:express",
  ],
)

cc_library(
  name = "elementwise",
  srcs = ["elementwise.cc"],
  hdrs = ["elementwise.h"],
  deps = [
    ":index",
  ],
)

cc_library(
  name = "expression",
  srcs = [
    "expression.cc",
    "scalar-flt-sse.cc",
    "scalar-flt-avx.cc",
    "vector-flt-sse.cc",
    "vector-flt-avx128.cc",
    "vector-flt-avx256.cc",
    "vector-flt-avx512.cc",
    "scalar-int.cc",
    "vector-int-sse.cc",
    "vector-int-avx128.cc",
    "vector-int-avx256.cc",
  ],
  hdrs = ["expression.h"],
  deps = [
    ":index",
    "//sling/base",
    "//sling/myelin:compute",
    "//sling/myelin:express",
  ],
)

