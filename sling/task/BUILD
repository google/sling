package(default_visibility = ["//visibility:public"])

load("//tools:embed.bzl", "embed_data")

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
    "//sling/util:asset",
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

embed_data(
  name = "app",
  srcs = [
    "app/appicon.ico",
    "app/index.html",
    "app/dashboard.css",
    "app/dashboard.js",
    "app/digital-7.mono.ttf",
  ],
)

cc_library(
  name = "dashboard",
  srcs = ["dashboard.cc"],
  hdrs = ["dashboard.h"],
  deps = [
    ":app",
    ":job",
    "//sling/base",
    "//sling/base:perf",
    "//sling/file:embed",
    "//sling/http:http-server",
    "//sling/http:static-content",
    "//sling/util:mutex",
  ],
)

cc_library(
  name = "process",
  srcs = ["process.cc"],
  hdrs = ["process.h"],
  deps = [
    ":task",
    "//sling/util:mutex",
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
  name = "pipe-reader",
  srcs = ["pipe-reader.cc"],
  deps = [
    ":process",
    ":task",
    "//sling/stream:input",
    "//sling/stream:unix-file",
  ],
  alwayslink = 1,
)

cc_library(
  name = "message-printer",
  srcs = ["message-printer.cc"],
  deps = [
    ":task",
    "//sling/base",
  ],
  alwayslink = 1,
)

cc_library(
  name = "sharder",
  srcs = ["sharder.cc"],
  deps = [
    ":task",
    "//sling/base",
    "//sling/util:fingerprint",
  ],
  alwayslink = 1,
)

cc_library(
  name = "mapper",
  srcs = ["mapper.cc"],
  hdrs = ["mapper.h"],
  deps = [
    ":task",
    "//sling/base",
  ],
)

cc_library(
  name = "reducer",
  srcs = ["reducer.cc"],
  hdrs = ["reducer.h"],
  deps = [
    ":task",
    "//sling/base",
    "//sling/util:mutex",
  ],
)

cc_library(
  name = "identity",
  srcs = ["identity.cc"],
  deps = [
    ":task",
  ],
  alwayslink = 1,
)

cc_library(
  name = "sorter",
  srcs = ["sorter.cc"],
  deps = [
    ":task",
    "//sling/base",
    "//sling/file:recordio",
    "//sling/string:printf",
    "//sling/util:mutex",
  ],
  alwayslink = 1,
)

cc_library(
  name = "null-sink",
  srcs = ["null-sink.cc"],
  deps = [
    ":task",
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
    "//sling/nlp/document:annotator",
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

cc_library(
  name = "accumulator",
  srcs = ["accumulator.cc"],
  hdrs = ["accumulator.h"],
  deps = [
    ":reducer",
    ":task",
    "//sling/base",
    "//sling/string:numbers",
    "//sling/string:text",
    "//sling/util:fingerprint",
    "//sling/util:mutex",
  ],
  alwayslink = 1,
)

cc_library(
  name = "learner",
  srcs = ["learner.cc"],
  hdrs = ["learner.h"],
  deps = [
    ":process",
    ":task",
    "//sling/base",
    "//sling/myelin:learning",
    "//sling/myelin:profile",
    "//sling/util:mutex",
  ],
)
