package(default_visibility = ["//visibility:public"])

cc_library(
  name = "index",
  srcs = ["index.cc"],
  hdrs = ["index.h"],
  deps = [
    "//base",
    "//myelin:compute",
    "//myelin:express",
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

