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

"""Command-line flags"""

import argparse
import sling.pysling as api

# Command line flag arguments.
arg = argparse.Namespace()

# Initialize command-line argument parser.
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Post-processing hooks.
hooks = []

def define(*args, **kwargs):
  """Define command-line flag."""
  parser.add_argument(*args, **kwargs)

def hook(callback):
  """Add hook for post-processing flags."""
  hooks.append(callback)

def parse():
  """Parse command-line flags."""
  # Register all the C++ flags.
  flags = api.get_flags()
  for name, help, default in flags:
    if type(default) == bool:
      parser.add_argument("--" + name,
                          help=help,
                          default=default,
                          action="store_true")
    else:
      parser.add_argument("--" + name,
                          help=help,
                          type=type(default),
                          default=default,
                          metavar="VAL")

  # Parse command line flags.
  global arg
  parser.parse_args(namespace=arg)

  # Set C++ flags.
  current = vars(arg)
  for name, help, default in flags:
    value = current[name]
    if value != default: api.set_flag(name, value)

  # Call all the post-processing hooks.
  for callback in hooks: callback(arg)

# Standard command-line flags.
define("--data",
       help="data directory",
       default="local/data",
       metavar="DIR")

define("--corpora",
       help="corpus directory",
       metavar="DIR")

define("--workdir",
       help="working directory",
       metavar="DIR")

define("--repository",
       help="SLING git repository directory",
       default=".",
       metavar="DIR")

def post_process_flags(arg):
  if arg.corpora == None: arg.corpora = arg.data + "/corpora"
  if arg.workdir == None: arg.workdir = arg.data + "/e"

hook(post_process_flags)

