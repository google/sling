#!/bin/sh

bazel build -c opt \
  base:* \
  file:* \
  frame:* \
  myelin:* \
  myelin/kernel:* \
  myelin/generator:* \
  myelin/cuda:* \
  nlp/document:* \
  nlp/parser:* \
  nlp/parser/tools:* \
  nlp/parser/trainer:* \
  stream:* \
  string:* \
  util:* \
  web:* \

