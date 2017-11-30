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

cc_binary(
  name = "sempar-component-test",
  srcs = ["sempar-component-test.cc"],
  deps = [
    "//sling/base",
    "//sling/file:posix",
    "//sling/nlp/document:document",
    "//sling/nlp/document:document-source",
    "//sling/nlp/parser:parser-action",
    "//sling/nlp/parser/trainer:sempar-component",
    "//sling/nlp/parser/trainer:shared-resources",
    "//sling/string:strcat",
    "//third_party/syntaxnet:dragnn-ops",
    "//third_party/syntaxnet:syntaxnet",
  ],
)

