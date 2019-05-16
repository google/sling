# SLING Installation and Building

## Trying out the parser

If you just want to try out the parser on a pre-trained model, you can install
the wheel with pip and download a pre-trained parser model. On a Linux machine
with Python 3.5 you can install a pre-built wheel:

```
sudo pip3 install http://www.jbox.dk/sling/sling-2.0.0-cp35-none-linux_x86_64.whl
```
and download the pre-trained model:
```
wget http://www.jbox.dk/sling/caspar.flow
```
You can then use the parser in Python:
```
import sling

parser = sling.Parser("caspar.flow")

text = input("text: ")
doc = parser.parse(text)
print(doc.frame.data(pretty=True))
for m in doc.mentions:
  print("mention", doc.phrase(m.begin, m.end))
```

## Installation

If you want to train a parser or use SLING for C++ development, you need to
download the source code and build it.

First, clone the GitHub repository.

```shell
git clone https://github.com/google/sling.git
cd sling
```

Next, run the `seup.sh` script to set up the SLING development environment
and build the code:
```shell
./setup.sh
```

This will perform the following steps:
* Install missing package dependencies, notably GCC and Python 3.
* Install [Bazel](https://bazel.build/) which is used as the build system for
  SLING.
* Build SLING from source.
* Remove the Python 2.7 SLING pip package if it is installed.
* Set up link to the SLING development enviroment for SLING Python 3 API.

The parser trainer uses PyTorch for training, so it also needs to be installed:

```shell
sudo pip3 install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
```

## Building

Operating system: Linux<br>
Languages: C++ (gcc or clang), Python 3.5+, assembler<br>
CPU: Intel x64 or compatible<br>
Build system: Bazel<br>

You can use the `buildall.sh` script to build all the source code:

```shell
tools/buildall.sh
```

If you haven't run the `setup.sh` script already, you then need to link the
sling Python module directly to the Python source directory to use it in
"developer mode":

```shell
sudo ln -s $(realpath python) /usr/lib/python3/dist-packages/sling
```

**NOTE:**
* In case you are using an older version of GCC (< v5), you may want to comment
  out [this cxxopt](https://github.com/google/sling/blob/f8f0fbd1a18596ccfe6dbfba262a17afd36e2b5f/.bazelrc#L8) in .bazelrc.
* We currently do not support OSX, but you can check out
  [issue #189](https://github.com/google/sling/issues/189) for help on building
  on OSX.
* Similarly, we do not support Windows, but you can check out
  [issue #296](https://github.com/google/sling/issues/296) for help on
  building SLING on Windows Subsystem for Linux (WSL).

