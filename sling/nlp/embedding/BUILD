package(default_visibility = ["//visibility:public"])

cc_library(
  name = "embedding-model",
  srcs = ["embedding-model.cc"],
  hdrs = ["embedding-model.h"],
  deps = [
    "//sling/myelin:builder",
    "//sling/myelin:compute",
    "//sling/myelin:flow",
    "//sling/myelin:gradient",
    "//sling/myelin:learning",
  ],
)

cc_library(
  name = "word-embeddings",
  srcs = ["word-embeddings.cc"],
  deps = [
    ":embedding-model",
    "//sling/base",
    "//sling/base:perf",
    "//sling/file:recordio",
    "//sling/file:textmap",
    "//sling/frame:serialization",
    "//sling/myelin:builder",
    "//sling/myelin:compiler",
    "//sling/myelin:profile",
    "//sling/nlp/document",
    "//sling/task:process",
    "//sling/util:embeddings",
    "//sling/util:random",
    "//sling/util:thread",
    "//sling/util:unicode",
  ],
  alwayslink = 1,
)

cc_library(
  name = "fact-embeddings",
  srcs = ["fact-embeddings.cc"],
  deps = [
    ":embedding-model",
    "//sling/base",
    "//sling/file:textmap",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/myelin:builder",
    "//sling/myelin:compiler",
    "//sling/myelin:learning",
    "//sling/task:frames",
    "//sling/task:learner",
    "//sling/util:embeddings",
    "//sling/util:mutex",
    "//sling/util:random",
  ],
  alwayslink = 1,
)

cc_library(
  name = "fact-plausibility",
  srcs = ["fact-plausibility.cc"],
  deps = [
    "//sling/base",
    "//sling/file:textmap",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/myelin:builder",
    "//sling/myelin:compiler",
    "//sling/myelin:gradient",
    "//sling/myelin:learning",
    "//sling/task:frames",
    "//sling/task:learner",
    "//sling/util:random",
  ],
  alwayslink = 1,
)

cc_library(
  name = "plausibility-model",
  srcs = ["plausibility-model.cc"],
  hdrs = ["plausibility-model.h"],
  deps = [
    "//sling/base",
    "//sling/frame:object",
    "//sling/frame:serialization",
    "//sling/frame:store",
    "//sling/myelin:compiler",
    "//sling/myelin:compute",
    "//sling/myelin:flow",
    "//sling/nlp/kb:facts",
  ],
)

