# Tensorflow.

# This uses the Tensorflow C++ API from the Tensorflow Python 2.7 package. You
# need to install this package to use Tensorflow in this way:
#
# sudo apt-get install python-pip python-dev
# pip install tensorflow
# https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp27-none-linux_x86_64.whl

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
  name = "tensorflow",
  includes = [
    "include",
    "include/external/nsync/public",
  ],
  linkopts = [
    "-lpython2.7",
    "-l:/usr/local/lib/python2.7/dist-packages/tensorflow/python/_pywrap_tensorflow_internal.so",
  ],
)

