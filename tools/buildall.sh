#!/bin/sh

bazel build -c opt $* \
  sling/base:* \
  sling/file:* \
  sling/frame:* \
  sling/http:* \
  sling/myelin:* \
  sling/myelin/kernel:* \
  sling/myelin/generator:* \
  sling/myelin/cuda:* \
  sling/nlp/document:* \
  sling/nlp/embedding:* \
  sling/nlp/kb:* \
  sling/nlp/silver:* \
  sling/nlp/parser:* \
  sling/nlp/parser/tools:* \
  sling/nlp/wiki:* \
  sling/pyapi:* \
  sling/stream:* \
  sling/string:* \
  sling/task:* \
  sling/util:* \
  sling/web:* \
  tools:* \

