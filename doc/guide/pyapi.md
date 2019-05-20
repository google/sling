# SLING Python API

A number of components in SLING can be accessed through the Python SLING API.
You can install the SLING Python wheel using pip:
```
sudo -H pip3 install http://www.jbox.dk/sling/sling-2.0.0-py3-none-linux_x86_64.whl
```
or you can [clone the repo and build SLING from sources](install.md).

# Table of contents

* [Frame stores](#frame-stores)
* [Record files](#record-files)
* [Documents](#documents)
* [Parsing](#parsing)
* [Phrase tables](#phrase-tables)
* [Dates](#dates)
* [Miscellaneous](#miscellaneous)

# Frame stores

The SLING [frame store](../../sling/frame/) can be used from Python. See the
[SLING Frames Guide](frames.md) for an introduction to semantic frames and the
SLING frame store concepts.

SLING frames live in a store, so you create a new global store this way:
```
import sling
commons = sling.Store()
```
Loading frames into the store:
```
commons.load("data/nlp/schemas/meta-schema.sling")
commons.load("data/nlp/schemas/document-schema.sling")
```
Loading binary encoded frames in legacy encoding:
```
actions = commons.load("local/sempar/out/table", binary=True)
```
Freezing a store makes it read-only and allows local stores to be created based
on the store:
```
commons.freeze()
```
Looking up frames in the store:
```
name = commons['name']
```
Create a local store:
```
store = sling.Store(commons)
```
Frames in the global store are now accessible from the local store:
```
doc = store['document']
```
Role values for frames can be accessed as attributes:
```
print(doc.name)
```
or using indexing:
```
print(doc['name'])
```
You can also use a frame value to access roles:
```
print(doc[name])
```
You can test if a frame has a role:
```
if 'name' in doc: print("doc has 'name'")
if name in doc: print("doc has name")
```
You can iterate over all the named frames (i.e. frames with an `id:` slot)
in a store:
```
for f in store: print(f.id)
```
The `parse()` method can be used for adding new frames to the store:
```
f = store.parse('{a:10 b:10.5 c:"hello" d:{:thing} e:[1,2,3]}')
```
The `frame()` method can be used to create a new frame from a dictionary:
```
f = store.frame({'a': 10, 'b': 10.5, 'c': "hello"})
```
or a list of slot tuples:
```
f = store.frame([('a', 10), ('b': 10.5), ('c': "hello")])
```
or just an id string:
```
f = store.frame("frame_id")
```
Slots can be added or modified using attribute assignment:
```
f.c = "hello world"
```
or using index assignment:
```
f[name] = "The F frame"
f['a'] = 20
```
New slots can be added using the `append()` method:
```
f.d.append(name, "A thing")
```
Multiple slots can be added using the `extend()` method:
```
f.extend({'foo': 10, 'bar': 20})
```
or using a list of slot tuples:
```
f.extend([('foo', 10), ('bar': 20)])
```
All the slots in a frame can be iterated:
```
for name, value in f:
  print("slot", name,"=", value)
```
or just the roles with a particular name:
```
for r in doc('role'):
  print("doc role", r)
```
Frames can be encoded in text format with the `data()` method:
```
print(f.data())
```
and with indentation:
```
print(f.data(pretty=True))
```
or with binary encoding:
```
print(len(f.data(binary=True)))
```
Arrays can be created with the `array()` method:
```
a = store.array([1, 2, 3])
```
Arrays can also be created with nil values that can be assigned later:
```
a = store.array(3)
a[0] = 1
a[1] = 2
a[2] = 3
```
SLING arrays work much in the same way as Python lists except that they have
a fixed size:
```
print(len(a))
print(a[1])
for item in a: print(item)
```
Finally, a store can be save to a file in textual encoding:
```
store.save("/tmp/txt.sling")
```
or in binary encoding:
```
store.save("/tmp/bin.sling", binary=True)
```

## Record files

[Record files](../../sling/file/recordio.h) are files with variable-size records
each having a key and a value. Individual records are (optionally) compressed
and records are stored in chunks which can be read independently.
The default chunk size is 64 MB.

A `RecordReader` is used for reading records from a record file and supports
iteration over all the records in the record file:
```
import sling

recin = sling.RecordReader("test.rec")
for key,value in recin:
  print(key, value)
recin.close()
```
The `RecordReader` class has the following methods:
* `__init__(filename[, bufsize])`<br>
  Opens record file for reading.
* `close()`<br>
  Closes the record reader.
* `read()`<br>
  Reads next record and returns the key and value for the record.
* `tell()`<br>
  Returns the current file position in the record file.
* `seek(pos)`<br>
  Seek to new file position `pos` in record file.
* `rewind()`<br>
  Seeks back to beginning of record file.
* `done()`<br>
  Checks for end-of-file.

To write a record file, you can use a `RecordWriter`:
```
recout = sling.RecordWriter("/tmp/test.rec")
recout.write("key1", "value1")
recout.write("key2", "value2")
recout.close()
```

The `RecordWriter` class has the following methods:
* `__init__(filename, [bufsize], [chunksize], [compression], [index])`<br>
  Initialize record file writer.
* `close()`<br>
  Closes the record writer.
* `write(key, value)`<br>
  Writes record to record file.
* `tell()`<br>
  Returns the current file position in the record file.

A set of sharded record files can be treated as a key-value database store,
and you can use a `RecordDatabase` for looking up records by key. Record file
sets consisting of multiple files need to be sharded by key fingerprint. If the
`index` parameter is set to True when creating a record file, an internal index
will be generated for the record file. This speeds up random access using
the `lookup` method.

```
# Write records to indexed record file.
N=1000
writer = sling.RecordWriter("/tmp/test.rec", index=True)
for i in range(N):
  writer.write(str(i), "Data for record number " + str(i))
writer.close()

# Look up each record in record database.
db = sling.RecordDatabase("/tmp/test.rec")
for i in range(N):
  print(db.lookup(str(i)))
db.close()
```

The `RecordDatabase` class has the following methods:
* `__init__(filepattern, [bufsize], [cache])`<br>
  Opens a set of record files specified by `filepattern`.
* `close()`<br>
  Closes the record database.
* `lookup(key)`<br>
  Look up record by key in the record file set. If the record files are indexed,
  the index is used for looking up the record. Otherwise, a linear scan is used
  for finding a matching record, which can be slow for large files.


## Documents

A _SLING document_ is a SLING frame formatted according to the
[document](../../data/nlp/schemas/document-schema.sling) schema. A document
has the raw text of the document as well as the tokens, mentions, and thematic
frames:
```
{
  :document
  text: "John loves Mary"
  tokens: [
    {word: "John" start: 0 size: 4},
    {word: "loves" start: 5 size: 5},
    {word: "Mary" start: 11 size: 4}
  ]
  mention: {
    begin: 0
    evokes: {=#1 :/saft/person}
  }
  mention: {
    begin: 1
    evokes: {
      :/pb/love-01
      /pb/arg0: #1
      /pb/arg1: #2
    }
  }
  mention: {
    begin: 2
    evokes: {=#2 :/saft/person}
  }
}
```

The SLING Python API has wrapper classes for working with SLING documents, which
are more convenient to use than manipulating them directly using the frame API.

The `DocumentSchema` class keeps track of all the frame ids for document
role names. It is faster to create a document schema in the global store and
pass that as an argument when creating new documents because the document role
names only need to be resolved once and not each time a new document is created.

Example: read all document from a corpus:
```
import sling

commons = sling.Store()
docschema = sling.DocumentSchema(commons)
commons.freeze()

num_docs = 0
num_tokens = 0
corpus = sling.RecordReader("local/data/corpora/sempar/train.rec")
for _,rec in corpus:
  store = sling.Store(commons)
  doc = sling.Document(store.parse(rec), store, docschema)
  num_docs += 1
  num_tokens += len(doc.tokens)

print("docs:", num_docs, "tokens:", num_tokens)
```

Example: read text from a file and create a corpus of tokenized documents:
```
import sling

# Create global store for common definitions.
commons = sling.Store()
docschema = sling.DocumentSchema(commons)
commons.freeze()

# Open input file.
fin = open("local/news.txt", "r")

# Create record output writer.
fout = sling.RecordWriter("/tmp/news.rec")

recno = 0
for text in fin.readlines():
  # Create local store.
  store = sling.Store(commons)

  # Read text from input file and tokenize.
  doc = sling.tokenize(text, store=store, schema=docschema)

  # Add your frames and mentions here...

  # Update underlying frame for document.
  doc.update()

  # Write document to record file.
  fout.write(str(recno), doc.frame.data(binary=True))
  recno += 1

fin.close()
fout.close()
```

Example: write document with annotations for "John loves Mary":
```
import sling

# Create global store for common definitions.
commons = sling.Store()
docschema = sling.DocumentSchema(commons)

# Create global schemas.
isa = commons["isa"]
love01 = commons["/pb/love-01"]
arg0 = commons["/pb/arg0"]
arg1 = commons["/pb/arg1"]
person = commons["/saft/person"]

commons.freeze()

# Create record output writer.
fout = sling.RecordWriter("/tmp/john.rec")

# Add annotated "John loves Mary" example.
store = sling.Store(commons)
doc = sling.tokenize("John loves Mary", store=store, schema=docschema)
john = store.frame({isa: person})
mary = store.frame({isa: person})
loves = store.frame({isa: love01, arg0: john, arg1: mary})
doc.add_mention(0, 1).evoke(john)
doc.add_mention(1, 2).evoke(loves)
doc.add_mention(2, 3).evoke(mary)

# Note: One can also say doc.evoke_type(start, end, type) as a short-hand for:
# f = store.frame({isa: type})
# doc.add_mention(start, end).evoke(f)

doc.update()
fout.write("0001", doc.frame.data(binary=True))

fout.close()
```

The `Document` class has the following methods and properties:
* `__init__(frame=None, store=None, schema=None)`<br>
  Creates a new document. If `frame` is None, a new "blank" document is created.
  Otherwise, the frame is used to initialize the document. If `store` is None,
  a new store is created for the document. If `schema` is None, a new
  `DocumentSchema` is created for the store, but it is faster to pass in a
  pre-initialized document schema when creating new documents.
* `text` (r/w property)<br>
  Sets/gets the raw text for the document.
* `tokens` (r/o property)<br>
  Returns a list of tokens in the document.
* `mentions` (r/o property)<br>
  Returns a list of mentions in the document.
* `themes` (r/o property)<br>
  Returns a list of themes in the document.
* `add_token(text=None, start=None, length=None, brk=SPACE_BREAK)`<br>
  Adds token to the document.
* `add_mention(begin, end)`<br>
  Adds mention to the document.
* `add_theme(theme)`<br>
  Adds thematic frame to the document.
* `evoke(begin, end, frame)`<br>
  Adds a mention from [begin, end) and evokes `frame` from it. Returns `frame`.
* `evoke_type(begin, end, type)`<br>
  Adds a mention from [begin, end) and evokes a new frame of type `type` from
  it. Returns the newly built frame.
* `update()`<br>
  Updates the underlying document frame. The `update()` method needs to be
  called after tokens, mentions, or themes have been added to the document.
* `phrase(begin, end)`<br>
  Returns phrase text for a span of tokens.
* `refresh_annotations()`<br>
  Re-initializes the document object from the underlying frame.

The `Token` class has the following properties:
* `index` (r/w int property)<br>
  Gets/sets the index of the token.
* `text` (r/w string property)<br>
  Gets/sets text for token.
* `start` (r/w int property)<br>
  Gets/sets the start position in the raw text for the token.
* `length` (r/w int property)<br>
  Gets/sets the length (in bytes) of the token in the raw text.
* `end` (r/o int property)<br>
  Returns the end position (exclusive) of the token in the raw text.
* `brk` (r/w int property)<br>
  Gets/sets the break level for the token, i.e. the spacing between this token
  and the previous token. The following token break levels are supported:

    - `NO_BREAK` no white space between tokens
    - `SPACE_BREAK` white space between tokens (default)
    - `LINE_BREAK` new line between tokens
    - `SENTENCE_BREAK` token starts a new sentence
    - `PARAGRAPH_BREAK` token starts a new paragraph
    - `SECTION_BREAK` token starts a new section
    - `CHAPTER_BREAK` token starts a new chapter

The `Mention` class has the following methods and properties:
* `begin` (r/w int property)<br>
  Gets/sets the index of the first token in the mention.
* `length` (r/w int property)<br>
  Gets/sets the number of tokens in the mention.
* `end` (r/o int property)<br>
  Returns the index of the first token after the mention.
* `evokes()`<br>
  Returns a list of of frames evoked by this mention.
* `evoke(frame)`<br>
  Adds frame evoked by this mention.
* `evoke_type(type)`<br>
  Makes a frame of type `type` and evokes it from this mention.

The `Corpus` class can be used for iterating over a corpus of documents stored in
record files:
```
for document in sling.Corpus("local/data/e/wiki/en/documents@10.rec"):
  print(document.text)
```
This will create a global store with the document schema symbols and create
a local store for each document. If you have a global store you can use this
instead, but it needs to be frozen before iterating over the documents:
```
kb = sling.Store()
corpus = sling.Corpus("local/data/e/wiki/en/documents@10.rec", commons=kb)
kb.freeze()
for document in corpus:
  print(document.text)
```
### LEX format

While annotated documents can be created using the methods on the `Document`
class, it is sometime more convenient to use _LEX_ formatted text, which is a
light-weight frame annotation format for text. This is like normal plain text,
but you can add mentions with annotations to the text using special markup.
Mentions in the text are enclosed in square brackets, e.g.
`[John] [loves] [Mary]`. One or more frames evoked from a mention can be added
to the mention using a vertical bar in the mention,
e.g. `[John|{:/saft/person}]`. The frames can be assigned ids to reference the
frames from other frames, e.g. `[John|{=#1 :/saft/person}]`. The full
"John loves Mary" can be encoded in LEX format like this:
```
[John|{=#1 :/saft/person}] [loves|{:/pb/love-01 /pb/arg0: #1 /pb/arg1: #2}] [Mary|{=#2 :/saft/person}]
```
Stand-alone frames can also be added outside the mentions and then referenced in
the mentions:
```
[John|#1] [loves|#3] [Mary|#2]
{=#1 :/saft/person}
{=#2 :/saft/person}
{=#3 :/pb/love-01 /pb/arg0: #1 /pb/arg1: #2}

```
If a stand-alone frame is not evoked by any mention, it is added to the document
as a theme.

Mentions can also be nested:
```
[[New York|#1] [University|#2]|#3] {=#3 +Q49210 P276:{=#1 +Q60} P31:{=#2 +Q3918}}
```

A document can be created from LEX-encoded text using the `lex()` method:
* `doc = sling.lex(text)`<br>
  Creates a new store and returns a document with the annotated text.
* `doc = sling.lex(text, store=store)`<br>
  Returns a new document in the store with the annotated text.
* `doc = sling.lex(text, store=store, schema=docschema)`<br>
  Returns a new document with the annotated text using a pre-initialized
  document schema.

## Parsing

A document needs to be tokenized before parsing:
* `doc = sling.tokenize(text)`<br>
  Creates a new store and returns a document with the tokenized text.
* `doc = sling.tokenize(text, store=store)`<br>
  Returns a new document in the store with the tokenized text.
* `doc = sling.tokenize(text, store=store, schema=docschema)`<br>
  Returns a new document with the tokenized text using a pre-initialized
  document schema.

The SLING frame semantic parser can be loaded from a flow file:
```
import sling
parser = sling.Parser("sempar.flow")
```

After the parser has been loaded, it can be used for parsing text and adding
semantic annotations to the text:
* `doc = parser.parse(text)`<br>
  Tokenize and parse the text and return new document with text, tokens, and
  frame annotations.
* `doc = parser.parse(frame)`<br>
  Create document from frame and parse the document.
* `parser.parse(doc)`<br>
  Parse the tokens in the document and add semantic annotations.

## Phrase tables

A phrase table contains a mapping from _names_ to frames. A phrase table for
Wikidata entities is constructed by the `phrase-table` task, and can be used for
fast retrieval of all entities having a (normalized) name matching a phrase:
```
import sling

# Load knowledge base and phrase table.
kb = sling.Store()
kb.load("local/data/e/wiki/kb.sling")
names = sling.PhraseTable(kb, "local/data/e/wiki/en/phrase-table.repo")
kb.freeze()

# Lookup entities with name 'Annette Stroyberg'.
for entity in names.lookup("Annette Stroyberg"):
  print(entity.id, entity.name)

# Query all entities named 'Funen' with frequency counts.
for m in names.query("Funen"):
  print(m.count(), m.id(), m.item().name, "(", m.item().description, ")")
```

The `lookup()` and `query()` methods return the matches in decreasing
frequency order.

## Dates

Dates in the knowledge base can be encoded as integers, strings, or frames:
* Numbers with eight digits represent dates, e.g. `20180605` is June 05, 2018.
  Only dates after 1000 AD can be represented as integers. Numbers with six
  digits are used to represent months, e.g. `197001` means January 1970.
  Likewise, numbers with four digits are used for years, three digits represents
  decades, two digits for centuries, and one digit for millennia.
* String-based date values are in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601)
  format, e.g. `"+2013-05-01T00:00:00Z"` is May 1, 2013. Dates with lower
  precision can be made by padding the year with `*`, e.g `"196*"` is the
  [1960s decade](https://www.wikidata.org/wiki/Q35724).
* Knowledge base items can also be used as dates, e.g. `Q6927` is the
  [20th century](https://www.wikidata.org/wiki/Q6927). These frames are
  interpreted as dates by parsing the `point in time`
  ([P585](https://www.wikidata.org/wiki/Property:P585)) property of the item.

The Date class can be used for parsing date values from the knowledge base, e.g.
getting the date of birth ([P569](https://www.wikidata.org/wiki/Property:P569))
for Annette Stroyberg ([Q2534120](https://www.wikidata.org/wiki/Q2534120)):
```
entity = kb["Q2534120"]
dob = sling.Date(entity["P569"])
print(dob.year, dob.month, dob.day)
```

The `Date` class has the following properties and methods:
* `year` (r/w int property)<br>
  Gets/sets year. Year is  0 if date is invalid. Dates BCE is represented with
  negative year numbers.
* `month` (r/w int property)<br>
  Gets/sets month (1=January). Month is 0 if there is no month in date.
* `day` (r/w int property)<br>
  Gets/sets day of month (first day of month is 1). Day is 0 if there is no
  day in date.
* `precision` (r/w int property)<br>
  Dates can have different granularities:
    * `MILLENNIUM` if date represents a millennium.
    * `CENTURY` if date represents a century.
    * `DECADE` if date represents a decade.
    * `YEAR` if date represents a year.
    * `MONTH` if date represents a month.
    * `DAY` if date represents a day.
* `iso()`<br>
  Returns date in ISO 8601 format.
* `value()`<br>
  Convert date to numeric format if the date can be encoded as an integer.
  This can only be done for dates after 1000 AD. Otherwise the date is returned
  in ISO 8601 format. This can be used for updating date properties in the
  knowledge base.

A `Calendar` object can be used for converting `Date` objects to text and
backing off to more coarse-grained date representations. A `Calendar` object
is initialized from a knowledge base by using the
[calendar definitions](../../data/wiki/calendar.sling):
```
cal = sling.Calendar(kb)
dob = sling.Date(19361207)
```
The `Calendar` class has the following methods:
* `str(date)`<br>
  Returns a human-readable representation of the date, e.g. `cal.str(dob)`
  returns "December 7, 1936". The primary language of the knowledge base is
  used for the conversion.
* `day(date)`<br>
  Returns an item frame representing the day and month of the date, e.g.
  `cal.day(dob)` returns December 7
  ([Q2299](https://www.wikidata.org/wiki/Q2299)).
* `month(date)`<br>
  Returns an item frame representing the month of the date, e.g.
  `cal.month(dob)` returns December
  ([Q126](https://www.wikidata.org/wiki/Q126)).
* `year(date)`<br>
  Returns an item frame representing the year of the date, e.g.
  `cal.year(dob)` returns 1936
  ([Q18649](https://www.wikidata.org/wiki/Q18649)).
* `decade(date)`<br>
  Returns an item frame representing the decade of the date, e.g.
  `cal.decade(dob)` returns 1930s
  ([Q35702](https://www.wikidata.org/wiki/Q35702)).
* `century(date)`<br>
  Returns an item frame representing the century of the date, e.g.
  `cal.century(dob)` returns 20th century
  ([Q6927](https://www.wikidata.org/wiki/Q6927)).
* `millennium(date)`<br>
  Returns an item frame representing the millennium of the date, e.g.
  `cal.millennium(dob)` returns 2nd millennium
  ([Q25860](https://www.wikidata.org/wiki/Q25860)).

## Miscellaneous

You can log messages to the SLING logging module:
```
import sling.log as log

log.info("Informational message")
log.warning("Warning message")
log.error("Error message")
log.fatal("Fatal error")
```

The SLING [command line flag module](../../sling/base/flags.h) is integrated
with the Python flags module, so the SLING flags can be set though a standard
Python [argparse.ArgumentParser](https://docs.python.org/2/library/argparse.html).
Flags are defined using the `flags.define()` method, e.g.
```
import sling.flags as flags

flags.define("--verbose",
             help="output extra information",
             default=False,
             action='store_true')
```
The `flags.define()` function takes the same arguments as the standard Python
[add_argument()](https://docs.python.org/2/library/argparse.html#the-add-argument-method)
method. You can then access the flags as variables in the flags module, e.g.:
```
  if flags.verbose:
    print("verbose output...")
```

The flags parser must be initialized in the main method of your Python program:
```
if __name__ == '__main__':
  # Parse command-line arguments.
  flags.parse()
```

The WikiConverter class can convert
[Wikidata items in JSON format](https://www.mediawiki.org/wiki/Wikibase/DataModel/JSON)
to SLING frame notation.
```
import sling
from urllib.request import urlopen

store = sling.Store()
wikiconv = sling.WikiConverter(store)

qid = "Q1254"
url = "https://www.wikidata.org/wiki/Special:EntityData/" + qid + ".json"
json = urlopen(url).read()[len(qid) + 16:-2]

item = wikiconv.convert_wikidata(store, json)
print(item.data(pretty=True, utf8=True))
```
