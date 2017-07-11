#!/bin/sh

bazel build -c opt \
  base:* \
  file:* \
  frame:* \
  myelin:* \
  myelin/generator:* \
  myelin/kernel:* \
  nlp/document:* \
  nlp/parser:* \
  nlp/parser/trainer:* \
  stream:* \
  string:* \
  util:* \
  web:* \

