package(default_visibility = ["//visibility:public"])

cc_library(
  name = "parser-action",
  srcs = ["parser-action.cc"],
  hdrs = ["parser-action.h"],
  deps = [
    "//sling/base",
    "//sling/frame:object",
    "//sling/frame:store",
    "//sling/string:strcat",
    "//sling/util:fingerprint",
  ],
)

cc_library(
  name = "parser-state",
  srcs = ["parser-state.cc"],
  hdrs = ["parser-state.h"],
  deps = [
    ":parser-action",
    "//sling/base",
    "//sling/frame:object",
    "//sling/frame:store",
    "//sling/nlp/document",
  ],
)

cc_library(
  name = "action-table",
  srcs = ["action-table.cc"],
  hdrs = ["action-table.h"],
  deps = [
    ":parser-action",
    "//sling/base",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/frame:store",
  ],
)

cc_library(
  name = "roles",
  srcs = ["roles.cc"],
  hdrs = ["roles.h"],
  deps = [
    ":action-table",
    ":parser-state",
    "//sling/frame:object",
  ],
)

cc_library(
  name = "parser-features",
  srcs = ["parser-features.cc"],
  hdrs = ["parser-features.h"],
  deps = [
    ":parser-action",
    ":parser-state",
    ":roles",
    "//sling/base",
    "//sling/myelin:compute",
  ],
)

cc_library(
  name = "parser",
  srcs = ["parser.cc"],
  hdrs = ["parser.h"],
  deps = [
    ":parser-features",
    ":parser-state",
    ":roles",
    "//sling/base",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/myelin:compiler",
    "//sling/myelin:profile",
    "//sling/nlp/document",
    "//sling/nlp/document:lexical-encoder",
  ],
)

cc_library(
  name = "parser-annotator",
  srcs = ["parser-annotator.cc"],
  deps = [
    ":parser",
    ":multiclass-delegate",
    "//sling/nlp/document:annotator",
  ],
  alwayslink = 1,
)

cc_library(
  name = "frame-evaluation",
  hdrs = ["frame-evaluation.h"],
  srcs = ["frame-evaluation.cc"],
  deps = [
    "//sling/base",
    "//sling/file",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/nlp/document:document",
    "//sling/nlp/document:document-corpus",
    "//sling/string:strcat",
    "//sling/string:printf",
  ],
)

cc_library(
  name = "transition-generator",
  hdrs = ["transition-generator.h"],
  srcs = ["transition-generator.cc"],
  deps = [
    ":parser-action",
    "//sling/base",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/nlp/document:document",
  ],
)

cc_library(
  name = "parser-trainer",
  srcs = ["parser-trainer.cc"],
  hdrs = ["parser-trainer.h"],
  deps = [
    "frame-evaluation",
    ":parser-action",
    ":parser-features",
    ":roles",
    "//sling/base",
    "//sling/frame:store",
    "//sling/frame:serialization",
    "//sling/myelin:builder",
    "//sling/myelin:compiler",
    "//sling/myelin:gradient",
    "//sling/myelin:learning",
    "//sling/nlp/document:document",
    "//sling/nlp/document:document-corpus",
    "//sling/nlp/document:lexical-encoder",
    "//sling/nlp/document:lexicon",
    "//sling/task:learner",
    "//sling/util:mutex",
    "//sling/util:unicode",
  ],
)

cc_library(
  name = "caspar-trainer",
  srcs = ["caspar-trainer.cc"],
  deps = [
    ":parser-trainer",
    ":transition-generator",
    "//sling/base",
    "//sling/myelin:builder",
    "//sling/myelin:flow",
    "//sling/myelin:gradient",
    "//sling/myelin:learning",
    "//sling/nlp/document:document",
    "//sling/nlp/document:document-corpus",
  ],
  alwayslink = 1,
)

cc_library(
  name = "multiclass-delegate",
  srcs = ["multiclass-delegate.cc"],
  deps = [
    ":action-table",
    ":parser",
  ],
  alwayslink = 1,
)

