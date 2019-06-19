package(default_visibility = ["//visibility:public"])

cc_library(
  name = "flow",
  srcs = ["flow.cc"],
  hdrs = ["flow.h"],
  deps = [
    "//sling/base",
    "//sling/file",
    "//sling/string:printf",
  ],
)

cc_library(
  name = "compute",
  srcs = [
    "compute.cc",
    "macro-assembler.cc",
  ],
  hdrs = [
    "compute.h",
    "macro-assembler.h",
  ],
  deps = [
    ":flow",
    "//sling/base",
    "//sling/file",
    "//sling/string:printf",
    "//third_party/jit:assembler",
  ],
)

cc_library(
  name = "simd-assembler",
  srcs = ["simd-assembler.cc"],
  hdrs = ["simd-assembler.h"],
  deps = [
    ":compute",
    "//sling/base",
  ],
)

cc_library(
  name = "profile",
  srcs = ["profile.cc"],
  hdrs = ["profile.h"],
  deps = [
    ":compute",
    "//sling/base",
    "//sling/base:clock",
    "//sling/string:printf",
    "//third_party/jit:cpu",
  ],
)

cc_library(
  name = "graph",
  srcs = ["graph.cc"],
  hdrs = ["graph.h"],
  deps = [
    ":flow",
    "//sling/base",
    "//sling/file",
    "//sling/string:printf",
  ],
)

cc_library(
  name = "multi-process",
  srcs = ["multi-process.cc"],
  hdrs = ["multi-process.h"],
  deps = [
    ":compute",
    "//sling/base",
  ],
)

cc_library(
  name = "express",
  srcs = ["express.cc"],
  hdrs = ["express.h"],
  deps = [
    "//sling/base",
  ],
)

cc_library(
  name = "builder",
  srcs = ["builder.cc"],
  hdrs = ["builder.h"],
  deps = [
    ":flow",
    "//sling/base",
  ],
)

cc_library(
  name = "gradient",
  srcs = ["gradient.cc"],
  hdrs = ["gradient.h"],
  deps = [
    ":flow",
    ":builder",
    "//sling/base",
  ],
)

cc_library(
  name = "learning",
  srcs = ["learning.cc"],
  hdrs = ["learning.h"],
  deps = [
    ":compute",
    ":flow",
    ":builder",
    "//sling/base",
  ],
)

cc_library(
  name = "rnn",
  srcs = ["rnn.cc"],
  hdrs = ["rnn.h"],
  deps = [
    ":builder",
    ":compute",
    ":flow",
    ":gradient",
  ],
)

cc_library(
  name = "elf-linker",
  srcs = ["elf-linker.cc"],
  hdrs = ["elf-linker.h"],
  deps = [
    ":compute",
    "//sling/base",
    "//sling/util:elf-writer",
  ],
)

cc_library(
  name = "aot-linker",
  srcs = ["aot-linker.cc"],
  hdrs = ["aot-linker.h"],
  deps = [
    ":compute",
    "//sling/base",
    "//sling/file",
    "//sling/util:elf-writer",
  ],
)

cc_library(
  name = "compiler",
  srcs = ["compiler.cc"],
  hdrs = ["compiler.h"],
  deps = [
    ":compute",
    ":elf-linker",
    ":flow",
    ":graph",
    ":profile",
    "//sling/base",
    "//sling/base:perf",
    "//sling/myelin/cuda:cuda-runtime",
    "//sling/myelin/kernel:cuda",
    "//sling/myelin/kernel:dragnn",
    "//sling/myelin/kernel:mkl",
    "//sling/myelin/kernel:tensorflow",
  ],
)

cc_binary(
  name = "analyze",
  srcs = ["analyze.cc"],
  deps = [
    ":compute",
    ":elf-linker",
    ":flow",
    ":graph",
    ":profile",
    "//sling/base",
    "//sling/file:posix",
    "//sling/myelin/cuda:cuda-runtime",
    "//sling/myelin/kernel:cuda",
    "//sling/myelin/kernel:dragnn",
    "//sling/myelin/kernel:tensorflow",
  ],
)

cc_binary(
  name = "nnc",
  srcs = ["nnc.cc"],
  deps = [
    ":compiler",
    ":aot-linker",
    "//sling/base",
    "//sling/file:posix",
  ],
)

