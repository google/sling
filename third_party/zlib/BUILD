package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD/MIT-like license (for zlib)

cc_library(
  name = "zlib",
  srcs = [
    "upstream/adler32.c",
    "upstream/compress.c",
    "upstream/crc32.c",
    "upstream/deflate.c",
    "upstream/gzclose.c",
    "upstream/gzlib.c",
    "upstream/gzread.c",
    "upstream/gzwrite.c",
    "upstream/infback.c",
    "upstream/inffast.c",
    "upstream/inflate.c",
    "upstream/inftrees.c",
    "upstream/trees.c",
    "upstream/uncompr.c",
    "upstream/zutil.c",
  ],
  hdrs = [
    "upstream/crc32.h",
    "upstream/deflate.h",
    "upstream/gzguts.h",
    "upstream/inffast.h",
    "upstream/inffixed.h",
    "upstream/inflate.h",
    "upstream/inftrees.h",
    "upstream/trees.h",
    "upstream/zconf.h",
    "upstream/zlib.h",
    "upstream/zutil.h",
    "zlib.h",
  ],
  copts = [
    "-O3",
    "-Wno-unused-variable",
    "-Wno-implicit-function-declaration",
  ],
)

cc_library(
  name = "miniunzip",
  srcs = [
    "upstream/contrib/minizip/unzip.c",
    "upstream/contrib/minizip/ioapi.c",
  ],
  hdrs = [
    "upstream/contrib/minizip/unzip.h",
    "upstream/contrib/minizip/ioapi.h",
  ],
  deps = [
    ":zlib",
  ],
  copts = [
    "-O3",
    "-Wno-unused-variable",
    "-Wno-implicit-function-declaration",
  ],
)

