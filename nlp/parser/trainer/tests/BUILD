package(default_visibility = ["//visibility:public"])

cc_binary(
  name = "make-test-docs",
  srcs = ["make-test-docs.cc"],
  deps = [
    "//base",
    "//file:posix",
    "//frame:object",
    "//frame:store",
    "//nlp/document:document",
    "//nlp/document:token-breaks",
    "//nlp/parser/trainer:shared-resources",
    "//string:strcat",
  ],
)

cc_binary(
  name = "sempar-component-test",
  srcs = ["sempar-component-test.cc"],
  deps = [
    "//base",
    "//file:posix",
    "//nlp/document:document",
    "//nlp/document:document-source",
    "//nlp/parser:parser-action",
    "//nlp/parser/trainer:feature",
    "//nlp/parser/trainer:fixed-features",
    "//nlp/parser/trainer:link-features",
    "//nlp/parser/trainer:sempar-component",
    "//nlp/parser/trainer:shared-resources",
    "//string:strcat",
    "//third_party/syntaxnet:dragnn-ops",
    "//third_party/syntaxnet:syntaxnet",
  ],
)


