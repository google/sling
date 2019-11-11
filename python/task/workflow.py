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

"""Interface to SLING workflow task system"""

import datetime
import glob
import os
import re
import time

import sling
import sling.pysling as api
import sling.flags as flags
import sling.log as log

flags.define("--dryrun",
             help="build worflows but do not run them",
             default=False,
             action='store_true')

flags.define("--monitor",
             help="port number for task monitor (0 means no monitor)",
             default=6767,
             type=int,
             metavar="PORT")

flags.define("--logdir",
             help="directory where workflow logs are stored",
             default="local/logs",
             metavar="DIR")

# Input readers.
readers = {
  "records": "record-file-reader",
  "zip": "zip-file-reader",
  "store": "frame-store-reader",
  "textmap": "text-map-reader",
  "text": "text-file-reader",
}

# Output writers.
writers = {
  "records": "record-file-writer",
  "store": "frame-store-writer",
  "textmap": "text-map-writer",
  "text": "text-file-writer",
}

# Track if workflow system has been activated, i.e. any workflow has been run.
active = False

class Shard:
  """A shard is one part of a multi-part set."""
  def __init__(self, part, total):
    self.part = part
    self.total = total

  def __hash__(self):
    return hash(self.part)

  def __eq__(self, other):
    return other != None and self.part == other.part

  def __ne__(self, other):
    return not(self == other)

  def __repr__(self):
    return "[%d/%d]" % (self.part, self.total)


class Format:
  """Data format with a file format and a record format. The record format
  consists of a key and a value format."""
  def __init__(self, fmt=None, file=None, key=None, value=None):
    if fmt == None:
      self.file = file
      self.key = key
      self.value = value
    else:
      # Parse format specifier into file, key, and value formats.
      # <file>[/[<key>:]<value>].
      self.file = None
      self.key = None
      self.value = None
      slash = fmt.find('/')
      if slash != -1:
        self.file = fmt[:slash]
        colon = fmt.find(':', slash + 1)
        if colon != -1:
          self.key = fmt[slash + 1:colon]
          self.value = fmt[colon + 1:]
        else:
          self.value = fmt[slash + 1:]
      else:
        self.file = fmt

  def as_message(self):
    """Return format as a message format."""
    if self.file == "message": return self
    return Format(file="message", key=self.key, value=self.value)

  def __repr__(self):
    s = self.file if self.file != None else "*"
    if self.key != None or self.value != None:
      s += "/"
      if self.key != None: s += self.key + ":"
      s += self.value if self.value != None else "*"
    return s


class Resource:
  """A resource is an external input or output, e.g. a file."""
  def __init__(self, name, shard, format):
    self.name = name
    self.shard = shard
    self.format = format

  def __repr__(self):
    s = "Resource(" + self.name
    if self.shard != None: s += str(self.shard)
    if self.format != None: s += " as " + str(self.format)
    s += ")"
    return s


class Binding:
  """A binding binds a resource to a task input or output."""

  def __init__(self, name, resource):
    self.name = name
    self.resource = resource

  def __repr__(self):
    s = "Binding(" + self.name + " = "  + self.resource.name
    if self.resource.shard != None: s += str(self.resource.shard)
    if self.resource.format != None: s += " as " + str(self.resource.format)
    s += ")"
    return s


class Port:
  """A port connects a channel to a task source or sink."""
  def __init__(self, task, name, shard):
    self.task = task
    self.name = name
    self.shard = shard

  def __repr__(self):
    s = str(self.task)
    s += "." + self.name
    if self.shard != None: s += str(self.shard)
    return s


class Channel:
  """A channel is used for sending messages from a producer task to a consumer
  task."""
  def __init__(self, format, producer, consumer):
    self.format = format
    self.producer = producer
    self.consumer = consumer

  def __repr__(self):
    s = "Channel("
    if self.producer != None:
      s += str(self.producer)
    else:
      s += "*"
    s += " -> "
    if self.consumer != None:
      s += str(self.consumer)
    else:
      s += "*"
    if self.format != None: s += " as " + str(self.format)
    s += ")"
    return s


class Task:
  """A task is used for processing inputs from resources and messages from
  sources. It can output data to output resource and send messages to channel
  sinks."""
  def __init__(self, type, name, shard):
    self.type = type
    self.name = name
    self.shard = shard
    self.inputs = []
    self.outputs = []
    self.sources = []
    self.sinks = []
    self.params = {}
    self.annotators = []

  def attach_input(self, name, resource):
    """Attach named input resource(s) to task."""
    if isinstance(resource, list):
      for r in resource: self.inputs.append(Binding(name, r))
    else:
      self.inputs.append(Binding(name, resource))

  def attach_output(self, name, resource):
    """Attach named output resource(s) to task."""
    if isinstance(resource, list):
      for r in resource: self.outputs.append(Binding(name, r))
    else:
      self.outputs.append(Binding(name, resource))

  def connect_source(self, channel):
    """Connect channel to a named input source for the task."""
    self.sources.append(channel)

  def connect_sink(self, channel):
    """Connect channel to a named output sink for the task."""
    self.sinks.append(channel)

  def add_param(self, name, value):
    """Add configuration parameter to task."""
    if value is True: value = 1
    if value is False: value = 0
    self.params[name] = str(value)

  def add_params(self, params):
    """Add configuration parameters to task."""
    if params != None:
      for name, value in params.items():
        self.add_param(name, value)

  def add_annotator(self, name):
    """Add annotator to task."""
    self.annotators.append(name)

  def __repr__(self):
    s = self.name
    if self.shard != None: s += str(self.shard)
    return s


class Scope:
  """A scope is used for defined a name space for task names. It implements
  context management so you can use with statements to define name spaces."""
  def __init__(self, wf, name):
    self.wf = wf
    self.name = name
    self.prev = None

  def __enter__(self):
    self.prev = self.wf.scope
    self.wf.scope = self
    return self

  def __exit__(self, type, value, traceback):
    self.wf.scope = self.prev

  def prefix(self):
    """Returns the name prefix defined in the scope by concatenating all nested
    name spaces."""
    parts = []
    s = self
    while s != None:
      parts.append(s.name)
      s = s.prev
    return '/'.join(reversed(parts))


def format_of(input):
  """Get format from one or more channels or resources."""
  if isinstance(input, list):
    return input[0].format
  else:
    return input.format

def length_of(l):
  """Get number of elements in list or None for singletons."""
  return len(l) if isinstance(l, list) else None

class Workflow(object):
  """A workflow is used for running a set of tasks over some input producing
  some output. The tasks can be connected using channels. A workflow can run
  asynchronously and supports multi-threaded processing."""
  def __init__(self, name):
    self.name = name
    self.tasks = []
    self.channels = []
    self.resources = []
    self.task_map = {}
    self.resource_map = {}
    self.scope = None
    self.job = None

  def namespace(self, name):
    """Defines a name space for task names."""
    return Scope(self, name)

  def task(self, type, name=None, shard=None, params=None):
    """A new task to workflow."""
    if name == None: name = type
    if self.scope != None: name = self.scope.prefix() + "/" + name
    basename = name
    index = 0
    while (name, shard) in self.task_map:
      index += 1
      name = basename + "-" + str(index)
    t = Task(type, name, shard)
    if params != None: t.add_params(params)
    self.tasks.append(t)
    self.task_map[(name, shard)] = t
    return t

  def resource(self, file, dir=None, shards=None, ext=None, format=None):
    """Adds one or more resources to workflow. The file parameter can be a file
    name pattern with wild-cards, in which case it is expanded to a list of
    matching resources. The optional dir and ext are prepended and appended to
    the base file name. The file name can also be a sharded file name (@n),
    which is expanded to a list of resources, one for each shard. The general
    format of a file name is as follows: [<dir>]<file>[@<shards>][ext]"""
    # Recursively expand comma-separated list of files.
    if "," in file:
      resources = []
      for f in file.split(","):
        r = self.resource(f, dir=dir, shards=shards, ext=ext, format=format)
        if isinstance(r, list):
          resources.extend(r)
        else:
          resources.append(r)
      return resources

    # Convert format.
    if type(format) == str: format = Format(format)

    # Combine file name parts.
    filename = file
    if dir != None: filename = os.path.join(dir, filename)
    if shards != None: filename += "@" + str(shards)
    if ext != None: filename += ext

    # Check if filename is a wildcard pattern.
    filenames = []
    if re.search(r"[\*\?\[\]]", filename):
      # Match file name pattern.
      filenames = glob.glob(filename)
    else:
      m = re.match(r"(.*)@(\d+)(.*)", filename)
      if m != None:
        # Expand sharded filename.
        prefix = m.group(1)
        shards = int(m.group(2))
        suffix = m.group(3)
        for shard in range(shards):
          fn = "%s-%05d-of-%05d%s" % (prefix, shard, shards, suffix)
          filenames.append(fn)
      else:
        # Simple filename.
        filenames.append(filename)

    # Create resources.
    n = len(filenames)
    if n == 0:
      return None
    elif n == 1:
      key = (filenames[0], None, str(format))
      r = self.resource_map.get(key)
      if r == None:
        r = Resource(filenames[0], None, format)
        self.resource_map[key] = r
        self.resources.append(r)
      return r
    else:
      filenames.sort()
      resources = []
      for shard in range(n):
        key = (filenames[shard], str(Shard(shard, n)), str(format))
        r = self.resource_map.get(key)
        if r == None:
          r = Resource(filenames[shard], Shard(shard, n), format)
          self.resource_map[key] = r
          self.resources.append(r)
          resources.append(r)
      return resources

  def channel(self, producer, name="output", shards=None, format=None):
    """Adds one or more channels to workflow. The channel(s) are connected as
    sinks to the producer(s). If shards are specified, this creates a sharded
    set of channels."""
    if type(format) == str: format = Format(format)
    if isinstance(producer, list):
      channels = []
      for p in producer:
        if shards != None:
          for shard in range(shards):
            ch = Channel(format, Port(p, name, Shard(shard, shards)), None)
            p.connect_sink(ch)
            channels.append(ch)
            self.channels.append(ch)
        else:
          ch = Channel(format, Port(p, name, None), None)
          p.connect_sink(ch)
          channels.append(ch)
          self.channels.append(ch)
      return channels
    elif shards != None:
      channels = []
      for shard in range(shards):
        sink = Port(producer, name, Shard(shard, shards))
        ch = Channel(format, sink, None)
        producer.connect_sink(ch)
        channels.append(ch)
        self.channels.append(ch)
      return channels
    else:
      ch = Channel(format, Port(producer, name, None), None)
      producer.connect_sink(ch)
      self.channels.append(ch)
      return ch

  def connect(self, channel, consumer, sharding=None, name="input"):
    """Connect channel(s) to consumer task(s)."""
    multi_channel = isinstance(channel, list)
    multi_task = isinstance(consumer, list)
    if not multi_channel and not multi_task:
      # Connect single channel to single task.
      if channel.consumer != None: raise Exception("already connected")
      channel.consumer = Port(consumer, name, None)
      consumer.connect_source(channel)
    elif multi_channel and not multi_task:
      # Connect multiple channels to single task.
      shards = len(channel)
      for shard in range(shards):
        if channel[shard].consumer != None: raise Exception("already connected")
        port = Port(consumer, name, Shard(shard, shards))
        channel[shard].consumer = port
        consumer.connect_source(channel[shard])
    elif multi_channel and multi_task:
      # Connect multiple channels to multiple tasks.
      shards = len(channel)
      if len(consumer) != shards: raise Exception("size mismatch")
      for shard in range(shards):
        if channel[shard].consumer != None: raise Exception("already connected")
        port = Port(consumer[shard], name, None)
        channel[shard].consumer = port
        consumer[shard].connect_source(channel[shard])
    else:
      # Connect single channel to multiple tasks using sharder.
      if sharding == None: sharding = "sharder"
      sharder = self.task(sharding)
      self.connect(channel, sharder)
      self.connect(self.channel(sharder,
                                shards=len(consumer),
                                format=channel.format),
                   consumer,
                   name=name)

  def read(self, input, name=None, params=None):
    """Add readers for input resource(s). The format of the input resource is
    used for selecting an appropriate reader task for the format."""
    if isinstance(input, list):
      outputs = []
      shards = len(input)
      for shard in range(shards):
        format = input[shard].format
        if type(format) == str: format = Format(format)
        if format == None: format = Format("text")
        tasktype = readers.get(format.file)
        if tasktype == None: raise Exception("No reader for " + str(format))

        reader = self.task(tasktype, name=name, shard=Shard(shard, shards))
        reader.add_params(params)
        reader.attach_input("input", input[shard])
        output = self.channel(reader, format=format.as_message())
        outputs.append(output)
      return outputs
    else:
      format = input.format
      if type(format) == str: format = Format(format)
      if format == None: format = Format("text")
      tasktype = readers.get(format.file)
      if tasktype == None: raise Exception("No reader for " + str(format))

      reader = self.task(tasktype, name=name)
      reader.add_params(params)
      reader.attach_input("input", input)
      output = self.channel(reader, format=format.as_message())
      return output

  def write(self, producer, output, sharding=None, name=None, params=None):
    """Add writers for output resource(s). The format of the output resource is
    used for selecting an appropriate writer task for the format."""
    # Determine fan-in (channels) and fan-out (files).
    if not isinstance(producer, list): producer = [producer]
    if not isinstance(output, list): output = [output]
    fanin = len(producer)
    fanout = len(output)

    # Use sharding if fan-out is different from fan-in.
    if sharding == None and fanout != 1 and fanin != fanout:
      sharding = "sharder"

    # Create sharder if needed.
    if sharding == None:
      input = producer
    else:
      sharder = self.task(sharding)
      if fanin == 1:
        self.connect(producer[0], sharder)
      else:
        self.connect(producer, sharder)
      input = self.channel(sharder, shards=fanout, format=producer[0].format)

    # Create writer tasks for writing to output.
    writer_tasks = []
    for shard in range(fanout):
      format = output[shard].format
      if type(format) == str: format = Format(format)
      if format == None: format = Format("text")
      tasktype = writers.get(format.file)
      if tasktype == None: raise Exception("No writer for " + str(format))

      if fanout == 1:
        writer = self.task(tasktype, name=name)
      else:
        writer = self.task(tasktype, name=name, shard=Shard(shard, fanout))
      writer.attach_output("output", output[shard])
      writer.add_params(params)
      writer_tasks.append(writer)

    # Connect producer(s) to writer task(s).
    if isinstance(input, list) and len(input) == 1: input = input[0]
    if fanout == 1: writer_tasks = writer_tasks[0]
    self.connect(input, writer_tasks)

    return output

  def pipe(self, command, format=None, name=None):
    """Run command and pipe output to channel."""
    reader = self.task("pipe-reader", name, params={"command": command})
    if type(format) == str: format = Format(format)
    if format is None: format = Format("pipe/text")
    output = self.channel(reader, format=format.as_message())
    return output

  def collect(self, *args):
    """Return list of channels that collects the input from all the arguments.
    The arguments can be channels, resources, or lists of channels or
    resources."""
    channels = []
    for arg in args:
      if isinstance(arg, Channel):
        channels.append(arg)
      elif isinstance(arg, Resource):
        channels.append(self.read(arg))
      elif isinstance(arg, list):
        for elem in arg:
          if isinstance(elem, Channel):
            channels.append(elem)
          elif isinstance(elem, Resource):
            channels.append(self.read(elem))
          else:
            raise Exception("illegal element")
      else:
        raise Exception("illegal argument")
    return channels if len(channels) > 1 else channels[0]

  def parallel(self, input, threads=5, queue=None, name=None):
    """Parallelize input messages over thread worker pool."""
    workers = self.task("workers", name=name)
    workers.add_param("worker_threads", threads)
    if queue != None: workers.add_param("queue_size", queue)
    self.connect(input, workers)
    return self.channel(workers, format=format_of(input))

  def map(self, input, type=None, format=None, params=None, name=None):
    """Map input through processor."""
    # Use input format if no format specified.
    if format == None: format = format_of(input).as_message()

    # Create mapper.
    if type != None:
      mapper = self.task(type, name=name)
      mapper.add_params(params)
      reader = self.read(input)
      self.connect(reader, mapper)
      output = self.channel(mapper, format=format)
    else:
      output = self.read(input)

    return output

  def shuffle(self, input, shards=None):
    """Shard and sort the input messages."""
    if shards != None:
      # Create sharder and connect input.
      sharder = self.task("sharder")
      self.connect(input, sharder)
      pipes = self.channel(sharder, shards=shards, format=format_of(input))

      # Pipe outputs from sharder to sorters.
      sorters = []
      for i in range(shards):
        sorter = self.task("sorter", shard=Shard(i, shards))
        self.connect(pipes[i], sorter)
        sorters.append(sorter)
    else:
      sorters = self.task("sorter")
      self.connect(input, sorters)

    # Return output channel from sorters.
    outputs = self.channel(sorters, format=format_of(input))
    return outputs

  def reduce(self, input, output, type=None, params=None, name=None):
    """Reduce input and write reduced output."""
    if type == None:
      # No reducer (i.e. identity reducer), just write input.
      reduced = input
    else:
      reducer = self.task(type, name=name)
      reducer.add_params(params)
      self.connect(input, reducer)
      reduced = self.channel(reducer,
                             shards=length_of(output),
                             format=format_of(output).as_message())

    # Write reduce output.
    self.write(reduced, output, params=params)
    return reducer

  def mapreduce(self, input, output, mapper, reducer=None, params=None,
                format=None):
    """Map input files, shuffle, sort, reduce, and output to files."""
    # Determine the number of output shards.
    shards = length_of(output)

    # Mapping of input.
    mapping = self.map(input, mapper, params=params, format=format)

    # Shuffling of map output.
    shuffle = self.shuffle(mapping, shards=shards)

    # Reduction of shuffled map output.
    self.reduce(shuffle, output, reducer, params=params)
    return output

  def start(self):
    """Start workflow. The workflow will be run in the background, and the
    done() and wait() methods can be used to determine if the workflow has
    completed."""
    # Make sure all output directories exist.
    self.create_output_directories()

    # Create underlying job in task system.
    if self.job != None: raise Exception("job already running")
    self.job = api.Job(self, self.name)

    # Start job.
    global active
    active = True
    self.job.start()

  def wait(self, timeout=None):
    """Wait until workflow completes."""
    if self.job == None: return True
    if timeout != None:
      return self.job.wait_for(timeout)
    else:
      self.job.wait()
      return True

  def done(self):
    """Check if workflow is done."""
    if self.job == None: return True
    return self.job.done()

  def counters(self):
    """Return map of counters for the workflow."""
    if self.job == None: return None
    return self.job.counters()

  def dump(self):
    """Return workflow configuration."""
    s = ""
    for task in self.tasks:
      s += "task " + task.name
      if task.shard: s += str(task.shard)
      s += " : " + task.type
      s += "\n"
      for annotator in task.annotators:
        s += "  annotator " + annotator + "\n"
      for param in task.params:
        s += "  param " + param + " = " + task.params[param] + "\n"
      for input in task.inputs:
        s += "  input " + str(input) + "\n"
      for source in task.sources:
        s += "  source " +  str(source) + "\n"
      for output in task.outputs:
        s += "  output " +  str(output) + "\n"
      for sink in task.sinks:
        s += "  sink " +  str(sink) + "\n"
    for channel in self.channels:
      s += "channel " + str(channel) + "\n"
    for resource in self.resources:
      s += "resource " + str(resource) + "\n"
    return s

  def create_output_directories(self):
    """Create output directories for workflow."""
    checked = set()
    for task in self.tasks:
      for output in task.outputs:
        directory = os.path.dirname(output.resource.name)
        if directory in checked: continue
        if not os.path.exists(directory): os.makedirs(directory)
        checked.add(directory)

def register_task(name, cls):
  """Register task processor in task system."""
  api.register_task(name, cls)

def start_monitor(port):
  """Start task monitor."""
  api.start_task_monitor(port)

def stop_monitor():
  """Stop task monitor."""
  global active
  if active:
    log.info("sending final status to monitor")
    api.finalize_dashboard()

def statistics():
  """Stats for running and completed jobs."""
  return api.get_job_statistics()

def save_workflow_log(path):
  global active
  if not active: return False
  if path is None or len(path) == 0: return False
  if not os.path.exists(path): return False
  logfn = path + "/" + time.strftime("%Y%m%d-%H%M%S") + ".json"
  logfile = open(logfn, "w")
  logfile.write(statistics())
  logfile.close()
  log.info("workflow stats saved in " + logfn)
  return True

def run(wf):
  # In dryrun mode the workflow is just dumped without running it.
  if flags.arg.dryrun:
    print(wf.dump())
    return

  # Start workflow.
  wf.start()

  # Wait until workflow completes. Poll every second to make the workflow
  # interruptible.
  done = False
  while not done: done = wf.wait(1000)

def startup():
  # Start task monitor.
  if flags.arg.monitor > 0: start_monitor(flags.arg.monitor)

def shutdown():
  # Stop task monitor.
  if flags.arg.monitor > 0: stop_monitor()

  # Save log to log directory.
  save_workflow_log(flags.arg.logdir)

