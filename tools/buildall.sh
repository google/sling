#!/bin/sh

bazel build -c opt \
  base:* \
  file:* \
  frame:* \
  myelin:* \
  myelin/kernel:* \
  nlp/document:* \
  nlp/parser:* \
  nlp/parser/trainer:* \
  stream:* \
  string:* \
  util:* \
  web:* \

