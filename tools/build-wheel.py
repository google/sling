# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Create a Python Wheel for installing SLING API."""

# The Python wheel produced by this script can be installed with the following
# command:
#
#   sudo -H pip install /tmp/sling-2.0.0-py3-abi3-linux_x86_64.whl
#
# If you are developing the SLING system, it is convenient to just add a
# link to the SLING repository directly from the Python package directory
# instead:
#
#   sudo ln -s $(realpath python) /usr/lib/python3/dist-packages/sling

import os
import sys
import hashlib
import base64
import zipfile
import distutils.util;

def sha256_checksum(filename, block_size=65536):
  """ Compute SHA256 digest for file."""
  sha256 = hashlib.sha256()
  with open(filename, 'rb') as f:
    for block in iter(lambda: f.read(block_size), b''):
      sha256.update(block)
  return base64.urlsafe_b64encode(sha256.digest()).rstrip(b'=')

def sha256_content_checksum(data):
  """ Compute SHA256 digest for data content."""
  sha256 = hashlib.sha256()
  sha256.update(data)
  return base64.urlsafe_b64encode(sha256.digest()).rstrip(b'=')

# Python version.
pyversion = str(sys.version_info.major)
abi = "none"

# Wheel package information.
platform = distutils.util.get_platform().replace("-", "_")
tag = "py" + pyversion + "-" + abi + "-" + platform
package = "sling"
version = "2.0.0"
dist_dir = package + "-" + version + ".dist-info"
data_dir = package + "-" + version + ".data/purelib"
record_filename = dist_dir + "/RECORD"

wheel_dir = "/tmp"
wheel_basename = package + "-" + version + "-" + tag + ".whl"
wheel_filename = wheel_dir + "/" + wheel_basename

# Files to distribute in wheel.
files = {
  'bazel-bin/sling/pyapi/pysling.so': '$DATA$/sling/pysling.so',

  'python/__init__.py': '$DATA$/sling/__init__.py',
  'python/flags.py': '$DATA$/sling/flags.py',
  'python/log.py': '$DATA$/sling/log.py',

  'python/myelin/__init__.py': '$DATA$/sling/myelin/__init__.py',
  'python/myelin/builder.py': '$DATA$/sling/myelin/builder.py',
  'python/myelin/flow.py': '$DATA$/sling/myelin/flow.py',
  'python/myelin/nn.py': '$DATA$/sling/myelin/nn.py',
  'python/myelin/tf.py': '$DATA$/sling/myelin/tf.py',

  'python/nlp/__init__.py': '$DATA$/sling/nlp/__init__.py',
  'python/nlp/document.py': '$DATA$/sling/nlp/document.py',
  'python/nlp/parser.py': '$DATA$/sling/nlp/parser.py',

  'python/task/__init__.py': '$DATA$/sling/task/__init__.py',
  'python/task/corpora.py': '$DATA$/sling/task/corpora.py',
  'python/task/download.py': '$DATA$/sling/task/download.py',
  'python/task/silver.py': '$DATA$/sling/task/silver.py',
  'python/task/embedding.py': '$DATA$/sling/task/embedding.py',
  'python/task/wiki.py': '$DATA$/sling/task/wiki.py',
  'python/task/workflow.py': '$DATA$/sling/task/workflow.py',
}

# Create new wheel zip archive.
wheel = zipfile.ZipFile(wheel_filename, "w")
record = ""

# Write wheel metadata.
wheel_metadata_filename = dist_dir + "/WHEEL"
wheel_metadata = """Wheel-Version: 1.0
Root-Is-Purelib: false
Tag: $TAG$""".replace("$TAG$", tag).encode()

record += wheel_metadata_filename + ",sha256=" + \
          sha256_content_checksum(wheel_metadata).decode() + "," + \
          str(len(wheel_metadata)) + "\n"
wheel.writestr(wheel_metadata_filename, wheel_metadata)

# Write package metadata.
package_metadata_filename = dist_dir + "/METADATA"
package_metadata = """Metadata-Version: 2.0
Name: sling
Version: $VERSION$
Summary: SLING frame semantic parsing framework
Home-page: https://github.com/google/sling
Author: Google
Author-email: sling-team@google.com
License: Apache 2.0
Download-URL: https://github.com/google/sling/releases
Platform: UNKNOWN
Classifier: Programming Language :: Python
Classifier: Programming Language :: Python :: $PYVERSION$

Google SLING frame semantic parsing framework
"""
package_metadata = package_metadata.replace("$VERSION$", version)
package_metadata = package_metadata.replace("$PYVERSION$", pyversion)
package_metadata = package_metadata.encode()

record += package_metadata_filename + ",sha256=" + \
          sha256_content_checksum(package_metadata).decode() + "," + \
          str(len(package_metadata)) + "\n"
wheel.writestr(package_metadata_filename, package_metadata)

# Copy all files to wheel ZIP archive.
for source in files:
  # Get source and destination file names.
  destination = files[source]
  destination = destination.replace("$INFO$", dist_dir)
  destination = destination.replace("$DATA$", data_dir)
  print("Install", source, "as", destination)

  # Write entry to RECORD file.
  size = os.path.getsize(source)
  checksum = sha256_checksum(source.encode()).decode()
  record += destination + ",sha256=" + checksum + "," + str(size)  + "\n"

  # Add file to wheel zip archive.
  wheel.write(source, destination)

# Add RECORD file to wheel.
print("Add", record_filename)
wheel.writestr(record_filename, record)

# Done.
wheel.close()
print("Wheel written to", wheel_filename)

