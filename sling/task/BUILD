package(default_visibility = ["//visibility:public"])

cc_library(
  name = "message",
  srcs = ["message.cc"],
  hdrs = ["message.h"],
  deps = [
    "//sling/base",
  ],
)

cc_library(
  name = "environment",
  hdrs = ["environment.h"],
  deps = [
    "//sling/base",
  ],
)

cc_library(
  name = "task",
  srcs = ["task.cc"],
  hdrs = ["task.h"],
  deps = [
    ":environment",
    ":message",
    "//sling/base",
    "//sling/base:registry",
    "//sling/string:numbers",
    "//sling/string:printf",
  ],
)

cc_library(
  name = "job",
  srcs = ["job.cc"],
  hdrs = ["job.h"],
  deps = [
    ":environment",
    ":task",
    "//sling/base",
    "//sling/file",
    "//sling/string:numbers",
    "//sling/util:mutex",
    "//sling/util:threadpool",
  ],
)

cc_library(
  name = "process",
  srcs = ["process.cc"],
  hdrs = ["process.h"],
  deps = [
    ":task",
    "//sling/util:thread",
  ],
)

cc_library(
  name = "workers",
  srcs = ["workers.cc"],
  deps = [
    ":task",
    "//sling/util:threadpool",
  ],
  alwayslink = 1,
)

