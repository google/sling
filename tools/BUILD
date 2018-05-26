package(default_visibility = ["//visibility:public"])

cc_binary(
  name = "embed-data",
  srcs = ["embed-data.cc"],
  deps = [
    "//sling/util:elf-writer",
  ],
)

cc_binary(
  name = "codex",
  srcs = ["codex.cc"],
  deps = [
    "//sling/base",
    "//sling/file",
    "//sling/file:recordio",
    "//sling/file:posix",
    "//sling/frame",
    "//sling/stream:memory",
    "//sling/string:printf",
    "//sling/util:fingerprint",
  ],
)

