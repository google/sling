#!/bin/sh

bazel build -c opt \
  sling/myelin:* \
  sling/myelin/kernel:* \
  sling/myelin/generator:* \
  sling/myelin/cuda:* \

