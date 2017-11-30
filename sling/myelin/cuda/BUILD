package(default_visibility = ["//visibility:public"])

cc_library(
  name = "cuda-api",
  srcs = ["cuda-api.cc"],
  hdrs = ["cuda-api.h"],
  deps = [
    "//sling/base",
  ],
  linkopts = [
    "-ldl",
  ],
)

cc_library(
  name = "cuda",
  srcs = ["cuda.cc"],
  hdrs = ["cuda.h"],
  deps = [
    ":cuda-api",
    "//sling/base",
  ],
)

cc_library(
  name = "cuda-runtime",
  srcs = ["cuda-runtime.cc"],
  hdrs = ["cuda-runtime.h"],
  deps = [
    ":cuda",
    "//sling/base",
    "//sling/myelin:compute",
  ],
)

cc_library(
  name = "cuda-kernel",
  srcs = ["cuda-kernel.cc"],
  hdrs = ["cuda-kernel.h"],
  deps = [
    ":cuda",
    ":cuda-runtime",
    "//sling/base",
    "//sling/myelin:compute",
  ],
)

