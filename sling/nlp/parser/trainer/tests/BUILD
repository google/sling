package(default_visibility = ["//visibility:public"])

cc_binary(
  name = "make-test-docs",
  srcs = ["make-test-docs.cc"],
  deps = [
    "//sling/base",
    "//sling/file:posix",
    "//sling/frame:object",
    "//sling/frame:store",
    "//sling/nlp/document:document",
    "//sling/nlp/document:token-breaks",
    "//sling/nlp/parser/trainer:shared-resources",
    "//sling/string:strcat",
  ],
)
