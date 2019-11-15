# SLING Frames

## Introduction <a name="intro">

SLING has a framework for storing, inspecting, manipulating, and transporting
semantic frames compactly and efficiently. The frames can be used both for
linguistic annotations as well as knowledge representations. SLING is not tied
to any particular frame semantic theory or knowledge ontology, but allows you
to combine information from many different sources in a unified representation
and perform inference across these domains.

At the most basic technical level, a frame consists of a list of slots, where
each slot has a name (role) and a value. The slot values can be literals like
numbers and strings, or links to other frames. The frames essentially form a
graph where the frames are the (typed) nodes and the slots are the labeled
edges. The frames can also be viewed as a [feature structure](https://en.wikipedia.org/wiki/Feature_structure)
and unification can be used for induction of new frames from existing frames.
Frames can also be used for representing more basic data structures like a
C struct with fields, a protocol buffer,  or a record in a database.

This guide will use a [simple schema](#toy-schema) with persons, locations, and
organization as a toy example for modeling [frames](#toy-frames) for
characters from The Simpsons, e.g.:

```sling
{=/en/homer
  :/toy/person
  name: "Homer Simpson"
  /toy/person/age: 36
  /toy/person/place: /en/springfield
  /toy/person/spouse: /en/marge
  /toy/person/child: /en/bart
  /toy/person/child: /en/lisa
  /toy/person/child: /en/maggie
  /toy/person/employer: /en/snpp
}
```

## Frame Semantics <a name="frame-semantics">
(source: [Wikipedia](https://en.wikipedia.org/wiki/Frame_semantics_&#40;linguistics&#41;))

SLING is inspired by [frame semantics](http://www.icsi.berkeley.edu/pubs/ai/framesemantics76.pdf),
which is a theory of linguistic meaning developed by [Charles Fillmore](https://en.wikipedia.org/wiki/Charles_J._Fillmore)
(1929 – 2014) of University of California, Berkeley. Frame semantics relates
linguistic semantics to encyclopedic knowledge and the basic idea is that one
cannot understand the meaning of a word without access to all the essential
knowledge that relates to that word. For example, one would not be able to
understand the word "sell" without knowing anything about the situation of a
commercial transaction, which also involves, among other things, a seller, a
buyer, goods, payment, and the relations between these. Thus, a word evokes a
frame of semantic knowledge relating to the specific concept it refers to.

A semantic frame is a collection of facts that specify "characteristic features,
attributes, and functions of a denotatum, and its characteristic interactions
with things necessarily or typically associated with it." [Keith Alan, 2001]
A semantic frame can also be defined as a coherent structure of related concepts
that are related such that without knowledge of all of them, one does not have
complete knowledge of any one.

Frame semantics not only relates to individual concept, but can be expanded to
phrases, entities, grammatical constructions and other larger and more complex
linguistic and ontological units. Semantic frames can be used in information
modeling for constructing knowledge bases of world knowledge and common sense,
and frame semantics can also form the basis for reasoning about metaphors,
metonymy, actions, perspective, etc.

## Creating frames <a name="creating">

SLING frames live inside a `Store`. A store is a container that tracks the all
the frames that have been allocated in the store, and serves as a memory
allocation arena for the allocated frames. When making a new frame, you specify
the store that the frame should be allocated in. The frame will live in this
store until its store is deleted or the frame is garbage collected because there
are no references to it.

```c++
#include "sling/frame/objects.h"
using namespace sling;

Store store;
```

A frame consists of a list of slots where each slot has a name and a value. A
`Builder` object is used to create a new frame. The example frame from the
[Introduction](#intro) can be created in the following way:

```c++
Builder homer(&store);

homer.AddId("/en/homer");
homer.AddIsA("/toy/person");
homer.Add("name", "Homer Simpson");
homer.Add("/toy/person/age", 39);
homer.AddLink("/toy/person/place", "/en/springfield");
homer.AddLink("/toy/person/spouse", "/en/marge");
homer.AddLink("/toy/person/child", "/en/bart");
homer.AddLink("/toy/person/child", "/en/lisa");
homer.AddLink("/toy/person/child", "/en/maggie");
homer.AddLink("/toy/person/employer", "/en/snpp");

Frame homer_simpson = homer.Create();
```

Each store has a symbol table that maps symbols like `/en/homer` to the frame
that it is mapped to. The `AddId()` method can be used for assigning an id to a
frame. This will add an `id:` slot to the frame, and the name will be added to
the symbol table and bound to the frame when it is created. A frame doesn't
need to have an id, and some frames have multiple ids.

You can assign a type to the frame using the `AddIsA()` method e.g.
`/toy/person` in the example above. This will add an `isa:` slot to the frame to
indicate its type. A frame can have multiple types or can be untyped.

Slots are added to the frame using the `Add()` methods. This is an overloaded
method that allows you to set string, integer, boolean, and float values for
slots. In the example above, the `name` role is set to the string
"Homer Simpson", and `/toy/person/age` is set to the integer 39. A frame can
have multiple instance of the same slot name which can be used for encoding
one-to-many or many-to-many relationships, e.g. `/toy/person/child`.

A slot can refer to another frame. The `AddLink()` method can be used for adding
slots with references to other named frames, i.e. other frames that have ids. In
the example above, we add references to frames that don't exist yet, like
`/en/marge`. If the target frame doesn't exist, a *proxy frame* will be created
and bound to this id. A proxy frame is a placeholder for a frame that has not
yet been created. You can then later create a frame with this id, and this
will replace the proxy frame so the previously added links to the proxy frame
will now refer to the new frame.

When all the slots have been added to the builder, you can use the `Create()`
method to actually allocate the frame in the store. It returns a `Frame` object,
which is a reference to the newly created frame.  While the `Builder` class has
value semantics, the `Frame` class has reference semantics, i.e. it represents a
reference to a frame in the store rather than the frame itself, so it is more
similar to Java objects than regular C++ objects. For instance, when assigning
one `Frame` object to another it will just reference the same frame instead of
making a copy of the frame. The `Frame` will also lock the object in the store
to prevent it from being reclaimed by the garbage collector as long as the
`Frame` object is still alive.

## Connecting frames <a name="connecting">

After a frame has been created, it can be used as the value of a slot when
defining other frames, e.g.:

```c++
Builder marge(&store);
marge.AddId("/en/marge");
marge.Add("name", "Marge Simpson");
marge.Add("/toy/person/age", 33);
marge.AddLink("/toy/person/place", "/en/springfield");
marge.Add("/toy/person/spouse", homer_simpson);
marge.AddLink("/toy/person/child", "/en/bart");
marge.AddLink("/toy/person/child", "/en/lisa");
Frame marge_simpsons = marge.Create();
```

Here the `/toy/person/spouse` role for Marge is set to refer to the Homer frame
directly without using an id by using `Add()` instead of `AddLink()` and
specifying `homer_simpson` instead of the id `/en/homer`.

Since the Marge frame has the id `/en/marge`, the new frame will also replace
the proxy that was created for `/en/marge` in the previous example so the
`/toy/person/spouse` role for Homer will refer to the Marge frame:

```c++
Frame spouse = homer_simpson.GetFrame("/toy/person/spouse");
if (spouse == marge_simpsons) {
  std::cout << "Homer and Marge are married\n";
}
```

## Printing frames <a name="printing">

A frame can be converted to text format using the `ToText()` function, e.g.:

```c++
#include "sling/frame/serialization.h"

std::cout << ToText(marge_simpsons, 2);
```

The second parameter to `ToText()` is the indent. If this parameter is omitted,
the frame will be output without line breaks and indentation. With two-space
indentation, the output will look like this:

```sling
{
  =/en/marge
  :/toy/person
  name: "Marge Simpson"
  /toy/person/age: 33
  /toy/person/place: /en/springfield
  /toy/person/spouse: /en/homer
  /toy/person/child: /en/bart
  /toy/person/child: /en/lisa
}
```

## Inspecting frames <a name="inspecting">

If you have an id for a frame, you can create a `Frame` object to inspect the
role values of the frame:

```c++
Frame ms(&store, "/en/marge");
CHECK(ms.valid());
std::cout << "Age: " << ms.GetInt("/toy/person/age");
```

You can also iterate over all the slots in the frame. It is faster to lookup
the role handle ahead of time instead of resolving it inside the for loop.

```c++
Handle n_child = store.Lookup("/toy/person/child");
for (const Slot &s : ms) {
  if (s.name == n_child) {
    Frame child(&store, s.value);
    std::cout << "Child: " << child.GetString("name");
  }
}
```

## Updating frames <a name="updating">

The `Builder` class can also be used for updating a frame by initializing it
with the id or handle of the frame. This will initialize the builder with the
existing slots for the frame and the Add, Delete, and Set methods can be used
for modifying the frame.

```c++
Builder marge(marge_simpsons);
marge.Set("/toy/person/age", 34);
marge.AddLink("/toy/person/child", "/en/maggie");
marge.Update();
```

If you only need to update or add a single slot of a frame, you can use the
Set and Add methods on the `Frame` object. However, if you need to update two
or more slots, it is usually faster to use a `Builder` object for updating the
frame.

## Global stores <a name="global-stores">

A store can normally only be accessed from one thread at a time. Updating a
store from multiple threads concurrently without serializing access, e.g. with
a mutex, can lead to corruption of the store.

You can *freeze* a store by calling the `Freeze()` method on the store. This
will make the store read-only, and it is then safe to access it from multiple
threads concurrently. Such a store is called a *global store*. When the store
is frozen, it is cleaned up by first garbage collecting all unused objects and
the internal heaps are shrunk to remove any unused memory areas. After the
store has been frozen, the frames in the store can no longer be modified and
no new frames can be added to the store.

You can then create local stores on top of a global store:

```c++
Store global;
<<< initialize global store>>>
global.Freeze();

Store local1(&global);
Store local2(&global);
```

The global store then serves as an extension of the local store. The frames
in the local store can have reference to the frames in the global store. You
can create multiple local store on top of the same global store. Typically you
initialize one global store with all the shared information like schemas or
other static common knowledge data. You then create a local store for each
document/query/request that can contain the frame analysis for the
item being analyzed. This analysis can contain references to the frames in the
global store. By construction, you cannot have references from the global store
to the local store since the local stores cannot be created until the global
store has been frozen and then the global store can no longer be updated.

## Handles <a name="handles">

Normally you use `Frame` objects to keep references to frames in the store. The
references are tracked so frames in the store that are referenced directly or
indirectly by `Frame` objects will not be reclaimed by the garbage collector.

Internally, SLING represents references to frames using handles. A *handle* is a
32-bit integer with a special encoding. The `Handle` type is implemented as a
POD struct type wrapping the handle value. Besides referencing frames, a handle
can also be used for representing integers and floating-point values. There is
no explicit boolean handle type, so boolean values are represented as integers
where false is 0 and true is 1. The `Handle::Integer()`, `Handle::Float()` and
`Handle::Bool()` methods can be used for converting integers, floats, and
booleans into handle values.

While `Frame` objects are tracked, this is not the case for `Handle` objects so
special care should be taken when dealing with these. Any update to a store can
trigger a garbage collection, and if there are no tracked references to a frame
it will be reclaimed by the garbage collector. Frames with ids are registered
in the symbol table in the store which is tracked, so these frames will not be
reclaimed. Custom reference tracking can be implemented by specializing the
`External` class.

The `Handles` class can be used for arrays of handles that need to be tracked,
and similarly the `Slots` class can be be used for tracked slots which are
basically pairs of name and value handles. These are example of classes
specializing `External` to implement tracking of external references.

The `Handle` type has a custom hash function so they can be used in as keys in
hashed containers. For example, `HandleMap<T>` is a hash map with `Handle` keys
and `HandleSet` is a set of handles. Please note that these handles are not
tracked.

## Objects <a name="objects">

A store has one or more internal heaps where the frames are stored. These heaps
also contain other type of objects like string, symbols and arrays as well as
the symbol table. A `Frame` object references one frame in the store. It is a
subclass of `Object` which can reference any type of object in the store. The
`Object` class has methods for determining the type of the object, e.g.
`Object::IsFrame()`, `Object::IsString()`, and `Object::IsInt()`. Please refer
to `frame/object.h` for a complete list of methods.

```c++
Object f = FromText(store, "{=/en/usa :/toy/location name: \"United States\"}");
Object n = FromText(store, "1234");
Object a = FromText(store, "[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]");
if (f.IsFrame()) {
  Frame usa = f.AsFrame();
  std::cout << "Name: " << usa.GetString("name") << "\n";
}
if (n.IsInt()) {
  std::cout << "Number: " << n.AsInt() << "\n";
}
if (a.IsArray()) {
  Array primes = a.AsArray();
  std::cout << "Length: " << a.length() << "\n";
}
```

The following types of objects are supported:

* `Object`<br>
  This is a general reference to any type of object in a SLING store.
  It can be used for both handles where the value is encoded in the handle, e.g.
  integers and floats, as well as reference types like strings, frames, arrays,
  and symbols. The `Object` class has methods for inspecting the type and value
  of the handle as well as methods for casting to other more specific types. The
  `Object` class is the superclass for the other classes below.
* `Frame`<br>
  The `Frame` class allows you to get and set the role values for the
  frame. A `Frame` object can also be used as container for iterating the slots
  for the frame.
* `String`<br>
  A `String` object references a string in the store. Strings are
  allocated in the store heaps and are treated as immutable. You can get the
  value of a string using the `String::value()` method. The `String` class has
  a constructor for creating new strings.
* `Symbol`<br>
  The `Symbol` class is a reference to a symbol in the store.  A
  symbol links a name to a value. Symbols are part of the symbol table which is
  implemented as hash table with buckets of linked lists of symbol. The symbol
  has a reference to the symbol name and also contains a hash value for the
  name for fast symbol lookup. It also has a next pointer for linking the
  symbols in the map buckets. A symbol can either be bound or unbound. An
  unbound symbol has itself as the value and is just a symbolic name. A bound
  symbol can either be resolved or unresolved. A resolved symbol references
  another object, typically a frame, but an unresolved symbol points to a proxy
  object, which can later be replaced when the symbol is resolved.
* `Array`<br>
  An array is used for storing a list of values that can be accessed
  by index. Array elements can be any type of objects. You can get and set the
  individual elements. The `Array` class has a constructor that can be used for
  allocating a new array from a vector of handles.

## Name binding <a name="binding">

In the previous examples we used strings symbols for role names etc. This can be
expensive since these need to be looked up in the symbol table. It is often more
efficient to look these up in advance. You can use the `Store::Lookup()` method
for looking up frames in the symbol table and these can then be used instead of
the string symbols when setting and getting values:

```c++
Handle n_age = store.Lookup("/toy/person/age");
std::cout << "Homer age: " << homer_simpson.GetInt(n_age);
```

The `Store::Lookup()` method looks up the handle for the symbol name in the
symbol table. If the id does not exist, a proxy frame is created. You can look
up symbols in a global store that can later be used for local stores based on
this global store.

Alternatively, you can use `Name` objects to bind names to symbols. These
support static initialization and can be early bound to symbols. A `Name` object
can be used for setting and getting role values instead of string symbols or
handles. A `Name` object is assigned to a `Names` object. When the `Bind()`
method is called on the `Names` object all the names assigned to the `Names`
object will be looked up and resolved.

```c++
class Homer {
 public:
  Homer(Store *global) { CHECK(names_.Bind(global); }

  Frame Create(Store *store) {
    Builder homer(store);
    homer.AddId("/en/homer");
    homer.AddIsA(n_person_);
    homer.Add(n_name_, "Homer Simpson");
    homer.Add(n_age_, 39);
    homer.AddLink(n_place_, "/en/springfield");
    homer.AddLink(n_spouse_, "/en/marge");
    homer.AddLink(n_child_, "/en/bart");
    homer.AddLink(n_child_, "/en/lisa");
    homer.AddLink(n_child_, "/en/maggie");
    homer.AddLink(n_employer_, "/en/snpp");
    return homer.Create();
  }

 private:
  Names names_;
  Name n_person_{names_, "/toy/person"};
  Name n_name_{names_, "name"};
  Name n_age_{names_, "/toy/person/age"};
  Name n_place_{names_, "/toy/person/place"};
  Name n_spouse_{names_, "/toy/person/spouse"};
  Name n_child_{names_, "/toy/person/child"};
  Name n_employer_{names_, "/toy/person/employer"};
};
```

## Reading and writing frames in text format <a name="reading">

SLING frames are stored internally in a compact format but they can be converted
to text format using the `ToText()` function, see [Printing frames](#printing).
Frames are output as a list of pairs of slot names and values. Frames are
bracketed by curly braces. The slot name and value are separated by colon and
slots are separated by space and optionally a comma. Integers, floating-point
numbers, and string are output in the same format as C/C++. Strings are
delimited with double-quotes and are using the same escaping rules as in C/C++.
Symbols are output verbatim if they consist of letters, digits, slashes,
hyphens and underscores. Other characters are escaped with a backslash.

There are three reserved symbols that are output with a special syntax. The
`id` slots are preceded by an equal sign, e.g. `=/en/homer`. The `id` slots are
used for registering frames in the symbol table. The `isa` slots are preceded by
a colon, e.g. `:/toy/person` and are used for adding types to frames. Last, you
can use `is` slots to indicate subtyping in [schemas](#schemas) and these are
output with a preceding plus sign, e.g. `+/toy/entity`.

Symbol | Example        | Description
-------|----------------|-------------------------------------------
`id:`  | `=/en/homer`   | Frame id for registration in symbol table
`isa:` | `:/toy/person` | Frame type for typed frames
`is:`  | `+/toy/entity` | Base type for specialization of schemas

You can read frames in text format back into a store using the `FromText()`
function:

```c++
#include "sling/frame/object.h"
#include "sling/frame/serialization.h"
#include "sling/frame/store.h"

Store store;
Object obj = FromText(&store,
    "{=/en/springfield :/toy/location "
    "name: \"Springfield\" "
    "/toy/location/country: /en/usa}");
if (obj.valid() && obj.IsFrame()) {
  Frame springfield = obj.AsFrame();
  std::cout << "Name: " << springfield.GetString("name") << "\n";
}
```

The `FromText()` function will return the first object in the input string or
`nil` in case of an error. It can read all types of SLING objects, including
numbers, strings, symbols, and frames. You can use the `Object::valid()` method
to check if any object was returned and `Object::IsFrame()` to check that a
frame was read. The `Object::AsFrame()` method can be used to cast the return
value to a frame.

The `ToText()` and `FromText()` functions are wrappers around more general
classes for reading and writing frames in text format. SLING uses the zero-copy
stream interface from the protocol buffer library as the basic input/output
abstraction. The `Input` and `Output` classes in `stream/stream.h`
uses the zero-copy stream abstraction for providing input and output to SLING.
SLING objects are parsed using the `Reader` class, which takes its input from an
`Input` object. Likewise, the `Printer` class is used for converting frames to
text and it outputs the text to an `Output` object.

I/O interface        | SLING stream   | SLING text serialization
---------------------|----------------|-------------------------
sling::InputStream   | sling::Input   | sling::Reader
sling::OutputStream  | sling::Output  | sling::Printer

The `StringReader` and `FileReader` classes in
`frame/serialization.h` are utility classes for reading frames
from either strings or files. The `StringPrinter` and `FilePrinter` utility
classes can be used for writing frames to strings and files.

The id slots are used for making references between frames when reading them
into a store. This will add all the frames with ids to the symbol table. If this
is not desirable, you can instead use temporary ids with the form `#<number>`.
These ids are only used while reading the frame from the input, but this does
not add any id slots to the frames, e.g.:

```sling
{=#1 :/toy/person name: "Homer Simpson" /toy/person/spouse: #2}
{=#2 :/toy/person name: "Marge Simpson" /toy/person/spouse: #1}
```

A number of flags can be used to control which frames are output by the frame
printer:

Flag    | Default | Description
--------|---------|-----------------------------------------------------------
shallow | true    | output frames with public ids by reference
global  | false   | output frames in the global store by value
byref   | true    | output anonymous frames by reference using temporary ids

## Encoding and decoding frames in binary format <a name="encoding">

While frames in text format are human-readable, they can be quite verbose and
take up a lot of space. Alternatively, you can store frames in binary encoded
format. This is both more compact and it is faster to serialize frames to and
from binary format. The binary serialization wire format is described in
`frame/wire.h`. Strings, symbols names, and frames are varint
size-encoded and each symbol only needs to be serialized once. This requires
substantially fewer memory allocations and symbol table lookups when decoding.

The binary SLING encoder/decoder uses the same input/output streams as the
text reader/printer. The `StringDecoder` and `FileDecoder` classes in
`frame/serialization.h` are utility classes for decoding frames
in binary format from either strings or files. The `StringEncoder` and
`FileEncoder` utility classes can be used for writing frames in binary format
to strings and files.

I/O interface       | SLING stream  | SLING binary serialization
--------------------|---------------|---------------------------
sling::InputStream  | sling::Input  | sling::Decoder
sling::OutputStream | sling::Output | sling::Encoder

This example shows how you can binary encode a frame and read it into another
store:

```c++
#include "sling/frame/object.h"
#include "sling/frame/serialization.h"
#include "sling/frame/store.h"

Store source;
Homer homer(&source);
Frame homer_simpsons = homer.Create(&source);

StringEncoder encoder(&source)
encoder.Encode(homer_simpsons);
const string &encoded = encoder.buffer();

Store target;
StringDecoder decoder(&target)
Object doh = decoder.Decode(encoded);
```

The shallow, global, and byref flags can be used for controlling which frame are
output by the frame encoder. These flags have the same meaning as for text
serialization.

You can encode and decode a whole store at a time. The `Encoder::EncodeAll()`
method can be used for outputting all frames in the symbol table of a store.
This can be read into another store later using the `Decoder::DecodeAll()`
method which keeps decoding frames from the input until all the input has been
read.

## Schemas <a name="schemas">

You can assign types to frames by adding `isa:` slots to the frame. A frame can
have zero, one or more types. A frame type is called a *schema*. For an example
of a schema, see the [schema](#toy-schema) for the examples used in this guide.

Schemas are defined as frames with the type `schema`. This is a simple schema
definition of an organization:

```sling
{=/toy/organization :schema +/toy/entity
  name: "Organization"
  role: {=/toy/organization/place :slot
    name: "place"
    source: /toy/organization
    target: /toy/location
  }
  role: {=/toy/organization/leader :slot
    name: "leader"
    source: /toy/organization
    target: /toy/person
  }
  role: {=/toy/organization/employee :slot
    name: "employee"
    source: /toy/organization
    target: /toy/person
    multi: 1
  }
}
```

A schema usually always has an id (`=/toy/organization`). In this example, the
`/toy/organization` extends the `/toy/entity` schema which means that it
inherits all the types, roles and bindings from `/toy/entity`. A schema can
extend multiple other schemas allowing for multiple inheritance. A schema can
have a `name:` and a `description:`. These are mostly just for display purposes.

A schema defines a number of `role:` roles of type `slot`. These are the roles
that frames of this type can have according to the schema. Just like a schema
definition, a role definition has an id, and optionally name and description.
The id for a role usually has the source schema as a prefix. A role also defines
the `source:` and `target:` schemas for the role. If a frame can have multiple
instances of a role, the role can be marked with `multi: 1`.
Roles also support inheritance. Role inheritance is indicated by adding `is:`
slots to the role definition. An inherited role acts like an alias although its
target type can be specialized. In terms of [schema unification](#unification),
this adds an implicit binding between the role and the inherited role.

```sling
{=/toy/company :schema +/toy/organization
  name: "Company"
  role: {=/toy/company/company_name :slot +name
    name: "company name"
    source: /toy/company
    target: string
  }
}
```

A schema can also define `binding:` roles which are constraints between the
roles in the schema. The following binding types are supported:

```sling
binding: [ <path> equals <path> ]
binding: [ <path> equals self ]
binding: [ <path> assign <value> ]
binding: [ <path> hastype <type> ]
```

For example, you can express the constraint that the children of your spouse are
also your children (not always true in real life):

```sling
binding: [ /toy/person/child equals /toy/person/spouse /toy/person/child ]
```

The schema compiler can compile these bindings into feature structure templates
which can then be used for constructing frames according to the schema
definitions. Please refer to `sling/schemata.h` for details.

In light of schema and role definitions, we can now see what is going on when
defining a frame like:

```sling
{=/en/snpp
  :/toy/organization
  name: "Springfield Nuclear Power Plant"
  /toy/organization/place: /en/springfield
  /toy/organization/leader: /en/burns
  /toy/organization/employee: /en/homer
}
```

The value of the `isa:` role, `:/toy/organization`, which is defining the type
for this frame, is actually a reference to the schema for this type. This way
you have direct access to the schema definition for the frame and can use this
for processing the frame. Likewise, the role "name", e.g.
`/toy/organization/place`, is not just a symbolic name but is actually a
reference to the role definition of this role. In this way, frames, schemas, and
roles are tightly integrated and meta data is readily available for inference.
Since the role values like `/en/homer` are also translated into references to
other frames, the `/en/snpp` frame above consists entirely of references to
other frames for both the slot names and values, with the exception of the
value of the `name:` role which points to the string object
`"Springfield Nuclear Power Plant"`.

The meta schema for schemas are defined in
`data/sling/meta-schema.sl`. This defines the basic schemas
for making other schemas, as well as simple types like int, string, float, etc.
It also defines schema families and the catalog which can be used for
organization schemas by domain. Schemas have been defined for a number of
different domains. Each domain uses a reserved id prefix to avoid collisions:

Prefix     | Schema domain
-----------|-------------------------------
/cxn/      | Constructionary
/f/        | Framery
/fn/       | FrameNet
/g/        | Knowledge Graph MID
/kg/       | Knowledge Graph schema
/m/        | Freebase MID
/nc/       | Noun compounds
/proto/    | Protocol buffer message types
/pb/       | PropBank
/saft/     | Entity types
/schema/   | Schema families
/vn/       | VerbNet
/wn/       | WordNet

The `SchemaCompiler` can be used for pre-computing information needed for
efficient type inference and unification. The `SchemaCompiler::PreCompute()`
method will pre-compute the following information for schemas and add these as
extra roles for the schema definition:

* `ancestors:`<br>
  A list of all the schema types that this schema inherits from, directly or
  indirectly. This list includes the schema itself. This makes it fast to check
  if a frame is an instance of a certain schema type.
* `rolemap:`<br>
  A mapping of all inherited roles to their aliased roles in the schema. This is
  used during feature structure unification to prune aliased roles.
* `template:`<br>
  A [unification](#unification) template for the schema definition. Using
  pre-compiled schema templates is much more efficient for frame construction
  and projection that having to generate the unification template each time it
  is needed.
* `projections:`<br>
  A list of all projection mappings that have this schemas as the source schema.

## Feature structures and unification <a name="unification">

A feature structure is a directed graph, where each node represents a frame,
and the edges between nodes represents frame slots. Each node in the graph
defines a subgraph which can be considered a feature structure in itself. This
connection between graphs with nodes and edges and frames with slots gives rise
to a correspondence between feature structures and frames where one can be
converted to the other.

A `FeatureStructure` represents a whole graph as an array of slots, where
special index handles are used for encoding references between nodes. A feature
structure can be initialized from a frame in the object store, or a pre-compiled
template containing a complete graph. A feature structure can also be converted
to a set of frames representing the same graph. A feature structure can be
assigned a type (or types) by adding `isa:` slots to the node.

A feature structure can either be atomic or complex. An atomic feature structure
is regarded as a simple value, e.g. an integer or string. A frame is also
regarded as atomic if it has (non-local) identity. All other frames are
considered complex feature structures.

The primary operation on feature structures is unification. Unification is a
binary operation over two features structures, used for comparing and combining
the information in the two feature structures and can be thought of as a
way of merging two graphs and detecting merge conflicts. Unification either
returns a merged feature structure with the information from both feature
structures, or fails if they are incompatible. Unification preserves and
possibly adds information to the resulting feature structure (monotonicity).

Two atomic feature structures can be unified if they have the same value, or if
one or both are nil or empty. The result of the unification is the value itself.

Two complex feature structures can be unified if the values of all the common
slots can be unified. The result of the unification is then the unified values
of all the common slots plus the slots from the each that are not in common.

Feature structure types are unified according to a type system that defines the
subsumption relationship between types.

The `SchemaCompiler` in `frame/schemata.h` can produce a feature
structure template for a schema. The `SchemaFeatureStructure` uses the
inheritance relations between schemas to define a type system and a subsumption
relation for the schema system. The following schema information is used for
creating the features structure template for a schema:

* The `is:` slots include the the schema feature structure for parent schemas
  recursively so all constraints are inherited.
* The `role:` slots generate target type nodes and alias node for inherited
  roles.
* The `binding:` slots generate constraints depending on the binding type. The
  paths in the bindings are generated as path in the feature structure graph.
  The `equals` operator creates an alias node for the two path. The `assign`
  operator creates a value constraint for the path, and the `hastype` operator
  generates a type node for the path.

The `Schemata` class can be used for constructing frames from the schema
templates using the `Schemata::Construct()` method. The `Schemata` class can
also be used for projecting frames though mappings and doing subsumption
checking.

## Appendix A: Simpsons toy example <a name="toy-frames">

This guide uses examples from The Simpsons as examples of frames. This appendix
contains the complete set of frames for this toy frame set.

```sling
{=/en/homer
  :/toy/person
  name: "Homer Simpson"
  /toy/person/age: 36
  /toy/person/place: /en/springfield
  /toy/person/spouse: /en/marge
  /toy/person/child: /en/bart
  /toy/person/child: /en/lisa
  /toy/person/child: /en/maggie
  /toy/person/employer: /en/snpp
}

{=/en/marge
  :/toy/person
  name: "Marge Simpson"
  /toy/person/age: 34
  /toy/person/place: /en/springfield
  /toy/person/spouse: /en/homer
  /toy/person/child: /en/bart
  /toy/person/child: /en/lisa
  /toy/person/child: /en/maggie
}

{=/en/bart
  :/toy/person
  name: "Bart Simpson"
  /toy/person/age: 10
  /toy/person/place: /en/springfield
  /toy/person/parent: /en/homer
  /toy/person/parent: /en/marge
}

{=/en/lisa
  :/toy/person
  name: "Lisa Simpson"
  /toy/person/age: 8
  /toy/person/place: /en/springfield
  /toy/person/parent: /en/homer
  /toy/person/parent: /en/marge
}

{=/en/maggie
  :/toy/person
  name: "Maggie Simpson"
  /toy/person/age: 1
  /toy/person/place: /en/springfield
  /toy/person/parent: /en/homer
  /toy/person/parent: /en/marge
}

{=/en/springfield
  :/toy/location
  name: "Springfield"
  /toy/location/country: /en/usa
}

{=/en/usa
  :/toy/location
  name: "United States"
}

{=/en/snpp
  :/toy/organization
  name: "Springfield Nuclear Power Plant"
  /toy/organization/place: /en/springfield
  /toy/organization/leader: /en/burns
  /toy/organization/employee: /en/homer
}

{=/en/burns
  :/toy/person
  name: "Montgomery Burns"
  /toy/person/place: /en/springfield
  /toy/person/employer: /en/snpp
}
```

([source](http://animatedtv.about.com/cs/faqs/f/faqsimpold.htm))

## Appendix B: Simpsons schema  <a name="toy-schema">

This appendix defines the schemas for the toy example in this guide. It defines
schemas for persons, locations, and organizations, which are all subcases of
entities.

```sling
{=/toy/entity :schema +named name: "Entity"}

{=/toy/person :schema +/toy/entity
  name: "Person"
  role: {=/toy/person/age :slot
    name: "place"
    source: /toy/person
    target: int
  }
  role: {=/toy/person/place :slot
    name: "place"
    source: /toy/person
    target: /toy/location
  }
  role: {=/toy/person/parent :slot
    name: "parent"
    source: /toy/person
    target: /toy/person
    multi: 1
  }
  role: {=/toy/person/child :slot
    name: "child"
    source: /toy/person
    target: /toy/person
    multi: 1
  }
  role: {=/toy/person/spouse :slot
    name: "spouse"
    source: /toy/person
    target: /toy/person
  }
  role: {=/toy/person/employer :slot
    name: "employer"
    source: /toy/person
    target: /toy/organization
  }
}

{=/toy/location :schema +/toy/entity
  name: "Location"
  role: {=/toy/location/country :slot
    name: "country"
    source: /toy/location
    target: /toy/location
  }
}

{=/toy/organization :schema +/toy/entity
  name: "Organization"
  role: {=/toy/organization/place :slot
    name: "place"
    source: /toy/organization
    target: /toy/location
  }
  role: {=/toy/organization/leader :slot
    name: "leader"
    source: /toy/organization
    target: /toy/person
  }
  role: {=/toy/organization/employee :slot
    name: "employee"
    source: /toy/organization
    target: /toy/person
    multi: 1
  }
}
```
