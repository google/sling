# Compile embedded data files into ELF object files.

def _genembed_impl(ctx):
  # Generate arguments to the embedded data compiler.
  args = []
  for i in ctx.attr.srcs:
    args += [f.path for f in i.files.to_list()]

  # Run embedded data compiler.
  ctx.actions.run(
    inputs = ctx.files.srcs,
    outputs = [ctx.outputs.out],
    arguments = ["-o", ctx.outputs.out.path] + args,
    progress_message = "Embedding %s" % ctx.label.name,
    executable = ctx.executable._embed_data_compiler
  )

genembed = rule(
  implementation = _genembed_impl,
  attrs = {
    "srcs": attr.label_list(
      allow_files = True
    ),
    "_embed_data_compiler": attr.label(
      default = Label("//tools:embed-data"),
      cfg = "host",
      executable = True,
    ),
  },
  outputs = {
    "out": "%{name}.o"
  },
)

def embed_data(name, srcs):
  embed_pkg = genembed(
    name = name + "_genembed",
    srcs = srcs,
  )
  native.cc_library(
    name = name,
    srcs = [name + "_genembed"],
    alwayslink = True,
    linkstatic = True,
  )

