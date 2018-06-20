package(default_visibility = ["//visibility:public"])

cc_library(
  name = "pyapi",
  srcs = [
    "pyapi.cc",
    "pyarray.cc",
    "pybase.cc",
    "pydate.cc",
    "pyframe.cc",
    "pymisc.cc",
    "pyparser.cc",
    "pyphrase.cc",
    "pyrecordio.cc",
    "pystore.cc",
    "pytask.cc",
  ],
  hdrs = [
    "pyarray.h",
    "pybase.h",
    "pydate.h",
    "pyframe.h",
    "pymisc.h",
    "pyparser.h",
    "pyphrase.h",
    "pyrecordio.h",
    "pystore.h",
    "pytask.h",
  ],
  deps = [
    ":tasks",
    "//sling/base",
    "//sling/file",
    "//sling/file:recordio",
    "//sling/frame",
    "//sling/http:http-server",
    "//sling/nlp/document",
    "//sling/nlp/wiki:phrase-table",
    "//sling/nlp/document:document-tokenizer",
    "//sling/nlp/document:lex",
    "//sling/nlp/document:phrase-tokenizer",
    "//sling/nlp/kb:calendar",
    "//sling/nlp/parser",
    "//sling/task:dashboard",
    "//sling/task:job",
    "//sling/stream:file",
    "//sling/stream:unix-file",
  ],
  copts = [
    "-Wno-pmf-conversions",
    "-Wno-write-strings",
    "-Wno-invalid-offsetof",
  ],
)

cc_library(
  name = "tasks",
  deps = [
    "//sling/nlp/wiki:profile-aliases",
    "//sling/nlp/wiki:wikidata-importer",
    "//sling/nlp/wiki:wikipedia-importer",
    "//sling/nlp/wiki:wikipedia-documents",
    "//sling/nlp/wiki:name-table-builder",
    "//sling/nlp/wiki:phrase-table-builder",

    "//sling/task:frame-store-reader",
    "//sling/task:frame-store-writer",
    "//sling/task:sharder",
    "//sling/task:sorter",
    "//sling/task:record-file-reader",
    "//sling/task:record-file-writer",
    "//sling/task:text-file-reader",
    "//sling/task:text-file-writer",
    "//sling/task:text-map-reader",
    "//sling/task:text-map-writer",
    "//sling/task:workers",
  ],
)

cc_binary(
  name = "pysling.so",
  deps = [
    ":pyapi",
    "//sling/base",
    "//sling/file:posix",
  ],
  linkshared = 1,
  linkstatic = 1,
)

