# Zlib compression.
bind(
  name = "zlib",
  actual = "//third_party/zlib",
)

# Command line flags.
bind(
  name = "gflags",
  actual = "//third_party/gflags"
)

# Logging.
bind(
  name = "glog",
  actual = "//third_party/glog"
)

# Protocol buffers.
http_archive(
  name = "com_google_protobuf",
  urls = ["https://github.com/google/protobuf/archive/v3.3.0.zip"],
  strip_prefix = "protobuf-3.3.0",
)
http_archive(
  name = "com_google_protobuf_cc",
  urls = ["https://github.com/google/protobuf/archive/v3.3.0.zip"],
  strip_prefix = "protobuf-3.3.0",
)

