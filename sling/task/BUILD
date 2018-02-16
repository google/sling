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

cc_library(
  name = "text-file-reader",
  srcs = ["text-file-reader.cc"],
  deps = [
    ":process",
    ":task",
    "//sling/stream:file-input",
  ],
  alwayslink = 1,
)

cc_library(
  name = "text-file-writer",
  srcs = ["text-file-writer.cc"],
  deps = [
    ":task",
    "//sling/stream:file",
    "//sling/stream:output",
    "//sling/util:mutex",
  ],
  alwayslink = 1,
)

cc_library(
  name = "text-map-reader",
  srcs = ["text-map-reader.cc"],
  deps = [
    ":process",
    ":task",
    "//sling/stream:file-input",
  ],
  alwayslink = 1,
)

cc_library(
  name = "text-map-writer",
  srcs = ["text-map-writer.cc"],
  deps = [
    ":task",
    "//sling/stream:file",
    "//sling/stream:output",
    "//sling/util:mutex",
  ],
  alwayslink = 1,
)

cc_library(
  name = "record-file-reader",
  srcs = ["record-file-reader.cc"],
  deps = [
    ":process",
    ":task",
    "//sling/base",
    "//sling/file:recordio",
  ],
  alwayslink = 1,
)

cc_library(
  name = "record-file-writer",
  srcs = ["record-file-writer.cc"],
  deps = [
    ":task",
    "//sling/base",
    "//sling/file:recordio",
    "//sling/util:mutex",
  ],
  alwayslink = 1,
)

cc_library(
  name = "frames",
  srcs = ["frames.cc"],
  hdrs = ["frames.h"],
  deps = [
    ":task",
    "//sling/base",
    "//sling/frame",
    "//sling/stream:file",
    "//sling/stream:memory",
  ],
)

cc_library(
  name = "frame-store-reader",
  srcs = ["frame-store-reader.cc"],
  deps = [
    ":frames",
    ":process",
    ":task",
    "//sling/base",
    "//sling/frame",
    "//sling/stream:file",
  ],
  alwayslink = 1,
)

cc_library(
  name = "frame-store-writer",
  srcs = ["frame-store-writer.cc"],
  deps = [
    ":frames",
    ":task",
    "//sling/base",
    "//sling/frame",
    "//sling/stream:file",
    "//sling/util:mutex",
  ],
  alwayslink = 1,
)

cc_library(
  name = "documents",
  srcs = ["documents.cc"],
  hdrs = ["documents.h"],
  deps = [
    ":frames",
    "//sling/nlp/document",
  ],
)

cc_library(
  name = "rekey",
  srcs = ["rekey.cc"],
  deps = [
    ":task",
    ":frames",
    "//sling/base",
    "//sling/frame",
  ],
  alwayslink = 1,
)

