package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# IMPORTANT: Ensure that you have followed the instructions in README.first.
cc_library(
  name = "dragnn",
  includes = [
      "include1",
      "include2",
      "include3",
  ],
  linkopts = [
    "-lpython2.7",
    "-l:/usr/local/lib/python2.7/dist-packages/dragnn/python/dragnn_cc_impl.so",
  ],
)
