package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
  name = "memory",
  hdrs = ["memory.h"],
  deps = [
    "//base",
  ],
)

cc_library(
  name = "types",
  hdrs = ["types.h"],
  deps = [
    "//base",
  ],
)

cc_library(
  name = "code",
  srcs = ["code.cc"],
  hdrs = ["code.h"],
  deps = [
    ":memory",
    ":types",
    "//base",
  ],
)

cc_library(
  name = "cpu",
  srcs = ["cpu.cc"],
  hdrs = ["cpu.h"],
  deps = [
    "//base",
  ],
)

cc_library(
  name = "assembler",
  srcs = ["assembler.cc"],
  hdrs = [
    "assembler.h",
    "instructions.h",
    "registers.h",
  ],
  deps = [
    ":code",
    ":cpu",
    ":memory",
    ":types",
    "//base",
  ],
)

