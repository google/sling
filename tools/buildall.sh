#!/bin/sh

bazel build -c opt \
  sling/base:* \
  sling/file:* \
  sling/frame:* \
  sling/myelin:* \
  sling/myelin/kernel:* \
  sling/myelin/generator:* \
  sling/myelin/cuda:* \
  sling/nlp/document:* \
  sling/nlp/parser:* \
  sling/nlp/parser/tools:* \
  sling/nlp/parser/trainer:* \
  sling/pyapi:* \
  sling/stream:* \
  sling/string:* \
  sling/task:* \
  sling/util:* \
  sling/web:* \

