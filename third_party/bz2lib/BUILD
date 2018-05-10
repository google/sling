# Imported from http://bzip.org/downloads.html

licenses(["notice"])

cc_library(
  name = "bz2lib",
  visibility = ["//visibility:public"],
  srcs = [
    "blocksort.c",
    "bzlib.c",
    "compress.c",
    "crctable.c",
    "decompress.c",
    "huffman.c",
    "randtable.c",
  ],
  hdrs = [
    "bzlib.h",
    "bzlib_private.h",
  ],
  copts = [
    "-Wno-unknown-warning-option",
    "-Wno-unused-const-variable",
    "-Wno-unused-but-set-variable",
    "-Wno-unused-private-field",
    "-DBZ_NO_STDIO",
  ]
)

