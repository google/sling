package(default_visibility = ["//visibility:public"])

cc_library(
  name = "token-breaks",
  hdrs = ["token-breaks.h"],
)

cc_library(
  name = "document",
  srcs = ["document.cc"],
  hdrs = ["document.h"],
  deps = [
    ":fingerprinter",
    ":token-breaks",
    "//base",
    "//frame:object",
    "//frame:store",
    "//string:text",
  ],
)

cc_library(
  name = "text-tokenizer",
  srcs = ["text-tokenizer.cc"],
  hdrs = ["text-tokenizer.h"],
  deps = [
    ":token-breaks",
    "//base",
    "//string:ctype",
    "//string:text",
    "//util:unicode",
    "//web:entity-ref",
  ],
)

cc_library(
  name = "fingerprinter",
  srcs = ["fingerprinter.cc"],
  hdrs = ["fingerprinter.h"],
  deps = [
    "//base",
    "//string:text",
    "//util:fingerprint",
    "//util:unicode",
  ],
)

cc_library(
  name = "document-tokenizer",
  srcs = ["document-tokenizer.cc"],
  hdrs = ["document-tokenizer.h"],
  deps = [
    ":document",
    ":text-tokenizer",
    "//base",
    "//frame:object",
    "//frame:store",
    "//string:text",
  ],
)

