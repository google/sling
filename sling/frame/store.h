// Copyright 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SLING_FRAME_STORE_H_
#define SLING_FRAME_STORE_H_

#include <stdlib.h>
#include <atomic>
#include <functional>
#include <string>
#include <utility>

#include "sling/base/bitcast.h"
#include "sling/base/logging.h"
#include "sling/base/macros.h"
#include "sling/base/types.h"
#include "sling/string/text.h"

namespace sling {

// Basic low-level data types.
typedef uint8 Byte;
typedef uint32 Word;
typedef Byte *Address;

// A region is an allocated memory area. The region has a two parts. The space
// between base and end is used, and the part between end and limit is unused.
// The region supports byte-oriented memory addressing.
class Region {
 public:
  // Initializes empty region.
  Region() : base_(nullptr), end_(nullptr), limit_(nullptr) {}

  // Deallocates the memory for the region.
  ~Region() { free(base_); }

  // Resizes the memory region to the requested size. The size is the number of
  // bytes that the region can store. It can be used to make the region smaller,
  // but not smaller than the currently used portion of the region.
  void reserve(size_t bytes);

  // Allocate memory from the unused portion expanding the region if there is
  // not enough free space.
  Address alloc(size_t bytes);

  // Allocate memory from the unused portion doubling the size of the region
  // until enough free space is available for the request.
  Address expand(size_t bytes);

  // Mark whole region as unused.
  void reset() { end_ = base_; }

  // Returns the number of bytes used in the region.
  size_t size() const { return end_ - base_; }

  // Returns the total number of bytes in the region (used and free).
  size_t capacity() const { return limit_ - base_; }

  // Returns number of unused bytes in the region.
  size_t available() const { return limit_ - end_; }

  // Returns true if the region is full.
  bool full() const { return end_ == limit_; }

  // Returns true if the region is empty.
  bool empty() const { return end_ == base_; }

  // Returns the size (in byte) of the region between begin and end.
  static size_t size(void *begin, void *end) {
    return reinterpret_cast<Address>(end) - reinterpret_cast<Address>(begin);
  }

 protected:
  // Beginning of memory region. Points to first byte of the memory region.
  Address base_;

  // End of used portion. Points to first unused byte of the memory region.
  Address end_;

  // End of memory region. Points to first byte after memory region.
  Address limit_;

 private:
  DISALLOW_COPY_AND_ASSIGN(Region);
};

// A space is a typed memory region. While it is similar to a vector it does not
// have element construction and destruction semantics.
template<typename T> class Space : public Region {
 public:
  Space() {}

  // Returns the beginning of the region.
  T *base() const { return reinterpret_cast<T *>(base_); }

  // Returns the end of the used part of the region.
  T *end() const { return reinterpret_cast<T *>(end_); }

  // Returns the end of the memory region.
  T *limit() const { return reinterpret_cast<T *>(limit_); }

  // Returns the number of elements.
  int length() const { return end() - base(); }

  // Allocates space from the unused portion of the memory region. Returns
  // false if there is not enough space left in the region.
  bool consume(size_t bytes, T **ptr) {
    Address next = end_ + bytes;
    if (next > limit_) return false;
    *ptr = end();
    end_ = next;
    return true;
  }

  // Allocates n elements from the unused portion of the memory region expanding
  // it if needed.
  T *add(int n) {
    Address ptr = end_;
    Address next = ptr + n * sizeof(T);
    if (next > limit_) {
      ptr = expand(n * sizeof(T));
    } else {
      end_ = next;
    }
    return reinterpret_cast<T *>(ptr);
  }

  // Removes n elements from the end of the used portion.
  T *remove(int n) {
    DCHECK(end_ - n * sizeof(T) >= base_);
    return reinterpret_cast<T *>(end_ -= n * sizeof(T));
  }

  // Stack operations for pushing and popping elements.
  T *push() { return add(1); }
  void pop() { remove(1); }
  T *top() { return end() - 1; }

  // Sets the end of the used region.
  void set_end(T *end) { end_ = reinterpret_cast<Address>(end); }

  // Returns a pointer with a byte offset from a base pointer.
  static T *address(T *ptr, Word offset) {
    return reinterpret_cast<T *>(reinterpret_cast<Address>(ptr) + offset);
  }

  // Returns pointer to location in region.
  T *address(Word offset) const {
    return reinterpret_cast<T *>(base_ + offset);
  }

  // Returns byte offset of pointer in region.
  Word offset(T *ptr) const {
    return reinterpret_cast<Address>(ptr) - base_;
  }

  // Returns index of element in region.
  Word index(T *ptr) const {
    return ptr - base();
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(Space);
};

// Heap object alignment (must be power of two). All objects in the object heaps
// are aligned to 8 bytes boundaries.
const Word kObjectAlign = 8;

// Rounds up n to match the object alignment.
inline Word Align(Word n) {
  return (n + kObjectAlign - 1) & ~(kObjectAlign - 1);
}

// A handle is a reference to an object in a store. It is represented as a
// 32-bit offset (i.e. byte offset, not array index) into a handle table. Bit 1
// is always zero for heap object handles to distinguish them from integer and
// float values. Bit 0 indicates whether it is a handle into the global pool (0)
// or the local pool (1).
//
//    332222222222111111111100000000 00
//    109876543210987654321098765432 10
//   +------------------------------+--+
//   | global handle table index    |00| global heap object index
//   +------------------------------+--+
//   | local handle table index     |01| local heap object index
//   +------------------------------+--+
//   | 30-bit signed integer        |10| integer
//   +------------------------------+--+
//   | 30-bit floating point number |11| float
//   +------------------------------+--+
//   |1111111111 20-bit index       |11| index (not an integer)
//   +------------------------------+--+
//
// The handle class is implemented as a POD type to make it efficient to pass by
// value.
struct Handle {
  static const int kIntShift     = 2;   // integers are shifted two bits
  static const int kHandleBits   = 32;  // handles are 32-bit integers
  static const int kTagBits      = 2;   // the two lowest bits are tag bits

  static const Word kTagMask     = 0x00000003;  // bit mask for handle tag
  static const Word kIntTag      = 0x00000002;  // 30-bit signed integer
  static const Word kFloatTag    = 0x00000003;  // 30-bit floating point number

  static const Word kRef         = 0x00000000;  // reference bit
  static const Word kNumber      = 0x00000002;  // number bit
  static const Word kPoolMask    = 0x00000001;  // bit mask for global or local

  static const Word kGlobal      = 0x00000000;  // global object pool
  static const Word kLocal       = 0x00000001;  // local object pool

  static const Word kGlobalTag   = kRef | kGlobal;
  static const Word kLocalTag    = kRef | kLocal;

  static const Word kNil         = kGlobalTag | 0x00000000;
  static const Word kId          = kGlobalTag | 0x00000004;
  static const Word kIsA         = kGlobalTag | 0x00000008;
  static const Word kIs          = kGlobalTag | 0x0000000C;

  static const Word kZero        = kIntTag | (0 << kIntShift);
  static const Word kOne         = kIntTag | (1 << kIntShift);
  static const Word kFalse       = kZero;
  static const Word kTrue        = kOne;

  // Range for integer handles.
  static const int kMinInt       = -2147483648 >> kIntShift;
  static const int kMaxInt       = 2147483647 >> kIntShift;

  // Maximum number of handles (local or global).
  static const int kMaxHandles   = 1 << (kHandleBits - kTagBits);

  // Returns the tag bits for the value.
  Word tag() const { return bits & kTagMask; }

  // Value type checking.
  bool IsInt() const { return tag() == kIntTag; }
  bool IsFloat() const { return tag() == kFloatTag; }
  bool IsNumber() const { return (bits & kNumber) != 0; }
  bool IsRef() const { return (bits & kNumber) == 0; }
  bool IsGlobalRef() const { return tag() == kGlobalTag; }
  bool IsLocalRef() const { return tag() == kLocalTag; }

  // Value checking.
  bool IsNil() const { return bits == kNil; }
  bool IsId() const { return bits == kId; }
  bool IsIsA() const { return bits == kIsA; }
  bool IsIs() const { return bits == kIs; }
  bool IsFalse() const { return bits == kFalse; }
  bool IsTrue() const { return !IsFalse(); }
  bool IsZero() const { return bits == kZero; }
  bool IsOne() const { return bits == kOne; }

  // Returns value as integer.
  int AsInt() const {
    DCHECK(IsInt());
    return static_cast<int>(bits) >> kIntShift;
  }

  // Returns value as boolean.
  bool AsBool() const {
    DCHECK(IsInt());
    return IsTrue();
  }

  // Returns value as floating-point number.
  float AsFloat() const {
    DCHECK(IsNumber());
    return IsFloat() ? bit_cast<float>(bits & ~kTagMask) : AsInt();
  }

  // Returns raw handle value.
  Word raw() const { return bits; }

  // Returns the index of the object in the handle table.
  Word idx() const { return bits >> kTagBits; }

  // Returns the pool for the object (global or local).
  Word pool() const { return bits & kPoolMask; }

  // Returns the handle value without the tag bits.
  Word untagged() const { return bits & ~kTagMask; }

  // Constructs integer handle.
  static constexpr Handle Integer(int n) {
    return Handle{static_cast<Word>(n << kIntShift) | kIntTag};
  }

  // Constructs float handle.
  static Handle Float(float n) {
    return Handle{(bit_cast<Word>(n) & ~kTagMask) | kFloatTag};
  }

  // Constructs boolean handle.
  static constexpr Handle Bool(bool b) {
    return Handle{b ? kTrue : kFalse};
  }

  // Constructs object reference handle.
  static Handle Ref(Word idx, Word tag) {
    DCHECK_EQ(tag & ~kTagMask, 0);
    return Handle{(idx << kTagBits) | tag};
  }

  // Constructs float handle from bits.
  static Handle FromFloatBits(Word bits) {
    return Handle{(bits << kTagBits) | kFloatTag};
  }

  // Returns floating point number as bits.
  Word FloatBits() const { return bits >> kTagBits; }

  // You can use an index handle for representing "special" integers using
  // negative floating-point NaN values. These do not collide with the normal
  // integers in the handle encoding. There are 20 bits used for representing
  // index handles.
  static const Word kIndexMask = 0xFFC00000 | kFloatTag;

  // Checks if a handle is an index handle.
  bool IsIndex() const { return (bits & kIndexMask) == kIndexMask; }

  // Returns index handle value as an integer.
  int AsIndex() const {
    DCHECK(IsIndex());
    return (static_cast<int>(bits) & ~kIndexMask) >> kIntShift;
  }

  // Constructs index handle.
  static constexpr Handle Index(int n) {
    return Handle{static_cast<Word>(n << kIntShift) | kIndexMask};
  }

  // A signaling NaN is used as an error value for handles.
  static const Word kError = kFloatTag | 0xFFBFFF00;

  // Checks if a handle is an error handle.
  bool IsError() const { return bits == kError; }

  // Equality testing.
  bool operator ==(Handle other) const { return bits == other.bits; }
  bool operator !=(Handle other) const { return bits != other.bits; }

  // Integer operations.
  void Add(int n) {
    DCHECK(IsInt());
    bits += (n << kIntShift);
  }
  void Subtract(int n) {
    DCHECK(IsInt());
    bits -= (n << kIntShift);
  }
  void Increment() { Add(1); }
  void Decrement() { Subtract(1); }

  // Special handles.
  static constexpr Handle nil() { return Handle{kNil}; }
  static constexpr Handle id() { return Handle{kId}; }
  static constexpr Handle isa() { return Handle{kIsA}; }
  static constexpr Handle is() { return Handle{kIs}; }
  static constexpr Handle zero() { return Handle{kZero}; }
  static constexpr Handle one() { return Handle{kOne}; }
  static constexpr Handle error() { return Handle{kError}; }

  // Handle is represented as an 32-bit unsigned integer where the lower bits
  // are used as tag bits to encode the handle type.
  Word bits;
};

// Hash function for handles.
struct HandleHash {
  size_t operator()(const Handle handle) const {
    return handle.raw() >> Handle::kTagBits;
  }
};

// Forward declarations.
class Store;
class Snapshot;
struct StringDatum;
struct FrameDatum;
struct SymbolDatum;
struct ProxyDatum;
struct ArrayDatum;
struct MapDatum;

// Handle ranges are used for representing memory regions of handles that still
// need to be traversed and marked during garbage collection.
struct Range {
  bool empty() const { return begin == end; }
  Handle *begin;
  Handle *end;
};

// The object type is stored in the upper bits of the size field in the object
// preamble. The top-most bit is 1 for frames and 0 for other object types. For
// frames, the lower two bits of the type are used for encoding identifier
// information like whether the frame has an id and if the frame is a proxy.
const Word kSizeBits = 28;
const Word kSizeMask = (1 << kSizeBits) - 1;
const Word kTypeMask = 0xE0000000;
const Word kTypeShift = kSizeBits + 1;
const Word kMarkMask = 0x10000000;
const Word kObjectSizeLimit = (1 << kSizeBits);
const Word kMapSizeLimit = kObjectSizeLimit / sizeof(Handle);

// Types.
enum Type : Word {
  // Heap object types.
  STRING  = 0x0UL << kTypeShift,
  SYMBOL  = 0x1UL << kTypeShift,
  ARRAY   = 0x2UL << kTypeShift,
  INVALID = 0x3UL << kTypeShift,
  FRAME   = 0x4UL << kTypeShift,

  // Simple value types. These types are never stored in the heaps and are only
  // used for returning types for number-based handles.
  INTEGER = 0x1UL,
  FLOAT   = 0x2UL,
};

// Frame type flags.
enum FrameFlags : Word {
  PROXY   = 0x1UL << kTypeShift,  // frame is a proxy (i.e. only has an id slot)
  PUBLIC  = 0x2UL << kTypeShift,  // frame has an id
};

// All heap objects starts with an 8 byte preamble that contains the handle for
// the object, the object size, and the object type. The object type and the
// mark bit are stored in the upper bits of the size field.
//
//    33222222222211111111110000000000
//    10987654321098765432109876543210
//   +--------------------------------+
//   |FTTM      payload size          | F=frame TT=type M=mark
//   +--------------------------------+
//   |      object handle           TT| TT=tag
//   +--------------------------------+
//
// The F bit indicates if the object is a frame. The TT bits for frames encodes
// if the frame is a proxy and whether the frame has an id. The M bit is a mark
// bit used for marking live objects during garbage collection. The mark bit is
// also set for all objects in frozen heaps.
struct Datum {
  // Returns address of object payload after the object preamble.
  const Address payload() const {
    return reinterpret_cast<Address>(const_cast<Datum *>(this) + 1);
  }

  // Returns address of end of object, i.e. first byte after payload.
  const Address limit() const { return payload() + size(); }

  // Returns payload range.
  void range(Range *range) const {
    range->begin = reinterpret_cast<Handle *>(payload());
    range->end = reinterpret_cast<Handle *>(limit());
  }

  // Returns the size of the heap object (exclusive the preamble).
  Word size() const { return info & kSizeMask; }

  // Returns the full type of the object.
  Type typebits() const { return static_cast<Type>(info & kTypeMask); }

  // Returns the base type of the object.
  Type type() const { return (info & FRAME) != 0 ? FRAME : typebits(); }

  // Returns true if heap object is marked.
  bool marked() const { return (info & kMarkMask) != 0; }

  // Marks/unmarks heap object.
  void mark() { info |= kMarkMask; }
  void unmark() { info &= ~kMarkMask; }

  // Invalidate heap object by setting the type to INVALID.
  void invalidate() { info = (info & ~kTypeMask) | INVALID; }

  // Resize object.
  void resize(int size) { info = (info & ~kSizeMask) | size; }

  // Returns the next object in the heap.
  Datum *next() const {
    return reinterpret_cast<Datum *>(payload() + Align(size()));
  }

  // Object type checking.
  bool IsString() const { return typebits() == STRING; }
  bool IsArray() const { return typebits() == ARRAY; }
  bool IsSymbol() const { return typebits() == SYMBOL; }
  bool IsInvalid() const { return typebits() == INVALID; }
  bool IsFrame() const { return (info & FRAME) != 0; }
  bool IsProxy() const {
    return ((info & (FRAME | PROXY)) == (FRAME | PROXY));
  }

  // Only strings contain binary data. All other types have handles as payload.
  bool IsBinary() const { return IsString(); }

  // Type casting.
  StringDatum *AsString() {
    DCHECK(IsString());
    return reinterpret_cast<StringDatum *>(this);
  }
  const StringDatum *AsString() const {
    DCHECK(IsString());
    return reinterpret_cast<const StringDatum *>(this);
  }
  FrameDatum *AsFrame() {
    DCHECK(IsFrame());
    return reinterpret_cast<FrameDatum *>(this);
  }
  const FrameDatum *AsFrame() const {
    DCHECK(IsFrame());
    return reinterpret_cast<const FrameDatum *>(this);
  }
  SymbolDatum *AsSymbol() {
    DCHECK(IsSymbol());
    return reinterpret_cast<SymbolDatum *>(this);
  }
  const SymbolDatum *AsSymbol() const {
    DCHECK(IsSymbol());
    return reinterpret_cast<const SymbolDatum *>(this);
  }
  ArrayDatum *AsArray() {
    DCHECK(IsArray());
    return reinterpret_cast<ArrayDatum *>(this);
  }
  const ArrayDatum *AsArray() const {
    DCHECK(IsArray());
    return reinterpret_cast<const ArrayDatum *>(this);
  }
  MapDatum *AsMap() {
    DCHECK(IsArray());
    return reinterpret_cast<MapDatum *>(this);
  }
  const MapDatum *AsMap() const {
    DCHECK(IsArray());
    return reinterpret_cast<const MapDatum *>(this);
  }
  ProxyDatum *AsProxy() {
    DCHECK(IsProxy());
    return reinterpret_cast<ProxyDatum *>(this);
  }
  const ProxyDatum *AsProxy() const {
    DCHECK(IsProxy());
    return reinterpret_cast<const ProxyDatum *>(this);
  }

  Word info;    // size of heap object payload and type bits
  Handle self;  // handle for heap object
};

// The payload of a string object contains the string. The size field is the
// exact size of the object, although the actual size of the string object
// is aligned. The strings are not zero-terminated.
struct StringDatum : public Datum {
  // Returns pointer to string.
  char *data() { return reinterpret_cast<char *>(payload()); }
  const char *data() const { return reinterpret_cast<const char *>(payload()); }

  // Returns string as a Text.
  Text str() const { return Text(data(), size()); }

  // Compares this string to a string buffer.
  bool equals(Text other) const {
    if (size() != other.size()) return false;
    return memcmp(data(), other.data(), other.size()) == 0;
  }

  // Compares this string to another string.
  bool equals(const StringDatum &other) const {
    if (size() != other.size()) return false;
    return memcmp(data(), other.data(), other.size()) == 0;
  }
};

// A slot is a name and value pair.
struct Slot {
  // Initializes slot.
  Slot() : name(Handle::nil()), value(Handle::nil()) {}
  Slot(Handle n, Handle v) : name(n), value(v) {}

  // Assigns new name and value to slot.
  void assign(Handle n, Handle v) { name = n; value = v; }

  // Swaps slot with another slot.
  void swap(Slot *other) {
    Handle tmp;
    tmp = name;
    name = other->name;
    other->name = tmp;
    tmp = value;
    value = other->value;
    other->value = tmp;
  }

  Handle name;   // slot name
  Handle value;  // slot value
};

// A frame consists of an array of slots with names and values.
struct FrameDatum : public Datum {
  // Range of slots for object.
  const Slot *begin() const {
    return reinterpret_cast<const Slot *>(payload());
  }
  Slot *begin() {
    return reinterpret_cast<Slot *>(payload());
  }
  const Slot *end() const {
    return reinterpret_cast<const Slot *>(limit());
  }
  Slot *end() {
    return reinterpret_cast<Slot *>(limit());
  }

  // Returns the number of slots in the frame.
  int slots() const { return size() / sizeof(Slot); }

  // Finds first value of named slot.
  Handle get(Handle name) const {
    for (const Slot *slot = begin(); slot < end(); ++slot) {
      if (slot->name == name) return slot->value;
    }
    return Handle::nil();
  }

  // Checks if frame has named slot.
  bool has(Handle name) const {
    for (const Slot *slot = begin(); slot < end(); ++slot) {
      if (slot->name == name) return true;
    }
    return false;
  }

  // Checks if frame has a slot with name and value.
  bool has(Handle name, Handle value) const {
    for (const Slot *slot = begin(); slot < end(); ++slot) {
      if (slot->name == name && slot->value == value) return true;
    }
    return false;
  }

  // Checks if frame has isa: type.
  bool isa(Handle type) const {
    for (const Slot *slot = begin(); slot < end(); ++slot) {
      if (slot->name == Handle::isa() && slot->value == type) return true;
    }
    return false;
  }

  // Checks if frame has is: type.
  bool is(Handle type) const {
    for (const Slot *slot = begin(); slot < end(); ++slot) {
      if (slot->name == Handle::is() && slot->value == type) return true;
    }
    return false;
  }

  // Updates the public flag for frame.
  void AddFlags(Word flags) { info |= (flags & PUBLIC); }

  // Returns true if frame has an id.
  bool IsPublic() const { return (info & PUBLIC) != 0; }

  // Returns true if frame is anonymous, i.e. it has no ids.
  bool IsAnonymous() const { return (info & PUBLIC) == 0; }
};

// A symbol links a name to a value. Symbols are usually stored in maps which
// can be used for symbol lookup. The symbol also contains a hash value for the
// name for fast symbol lookup and a next pointer for linking the symbols in
// the map buckets. A symbol can either be bound or unbound. An unbound symbol
// has itself as the value and is just a symbolic name. A bound symbol can
// either be resolved or unresolved. A resolved symbol references another
// object, but an unresolved symbol points to a proxy object, which can later
// be replaced when the symbol is resolved, so this gives three kinds of
// symbols:
//   1) if the symbol is unbound the value is the symbol itself.
//   2) if the symbol is bound and resolved, the value is the bound object.
//   3) if the symbol is bound and unresolved, the value is the proxy.
struct SymbolDatum : public Datum {
  // A symbol is unbound if its value is the symbol itself.
  bool unbound() const { return value == self; }
  bool bound() const { return value != self; }

  // Size of payload of symbol object.
  static const int kSize = 4 * sizeof(Handle);

  Handle hash;   // hash value for name encoded as a tagged integer
  Handle next;   // pointer to next symbol in map bucket
  Handle name;   // symbol name; a string object with the name of the symbol
  Handle value;  // symbol value; either a frame, a proxy, or the symbol itself
};

// A proxy object is used as a stand-in for the value of an unresolved frame.
// Please notice that it has the same object layout as frame with one id slot
// with the symbol for the proxy.
struct ProxyDatum : public FrameDatum {
  // Size of payload of symbol object.
  static const int kSize = 2 * sizeof(Handle);

  Handle id;      // constant id handle
  Handle symbol;  // symbol that has this as a proxy
};

// An array is used for storing a list of values that can be accessed by index.
struct ArrayDatum : public Datum {
  // Range of elements for array.
  Handle *begin() const { return reinterpret_cast<Handle *>(payload()); }
  Handle *end() const { return reinterpret_cast<Handle *>(limit()); }

  // Returns the number of elements in the array.
  int length() const { return size() / sizeof(Handle); }

  // Returns pointer to element in the array.
  Handle *at(int index) const {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, length());
    return begin() + index;
  }

  // Returns element in the array.
  Handle get(int index) const { return *at(index); }
};

// A map is an array used for storing symbols. It is implemented as a hash table
// with an array of buckets pointing to linked lists of symbols. Each linked
// list is terminated by nil.
struct MapDatum : public ArrayDatum {
  // Returns a pointer to the bucket for the hash value.
  Handle *bucket(Handle hash) const {
    DCHECK(hash.IsInt());
    return reinterpret_cast<Handle *>(payload() + (hash.untagged() % size()));
  }

  // Inserts symbol in map.
  void insert(SymbolDatum *symbol) {
    Handle *b = bucket(symbol->hash);
    symbol->next = *b;
    *b = symbol->self;
  }
};

// A root is an external handle that is tracked by the GC. Roots are used for
// holding on to objects in the heap so they are not garbage collected.
class Root {
 public:
  // Initializes nil root object.
  Root() : handle_(Handle::nil()), prev_(this), next_(this) {}

  // Initializes untracked handle.
  explicit Root(Handle handle) : handle_(handle), prev_(this), next_(this) {}

  // Adds root to store.
  Root(Store *store, Handle handle);

  // Removes root from store.
  ~Root() {
    prev_->next_ = next_;
    next_->prev_ = prev_;
  }

  // Checks if root is locked, i.e. it is linked into a root list.
  bool locked() const { return prev_ != this; }

 protected:
  friend class Store;

  // Initialize root.
  void InitRoot(Store *store, Handle handle);

  // Link root into root list.
  void Link(const Root *list) {
    prev_ = list;
    next_ = list->next_;
    list->next_->prev_ = this;
    list->next_ = this;
  }

  // Unlink root from root list.
  void Unlink() {
    prev_->next_ = next_;
    next_->prev_ = prev_;
    prev_ = next_ = this;
  }

  // Tracked handle.
  Handle handle_;

  // Roots are kept in a double-linked circular list.
  mutable const Root *prev_;
  mutable const Root *next_;
};

// External objects can hold references to objects in the store and will be
// called during GC to mark the active objects. In contrast to root objects
// which only track one object, an external object can track a dynamic array of
// objects.
class External {
 public:
  explicit External(Store *store);
  virtual ~External();

  // The external object must store the references in a contiguous range between
  // begin and end.
  virtual void GetReferences(Range *range) {
    range->begin = range->end = nullptr;
  }

 protected:
  friend class Store;

  // Constructor for sentinel in store.
  External();

  // Unlink object from list.
  void Unlink() {
    prev_->next_ = next_;
    next_->prev_ = prev_;
  }

  // External objects are kept in a double-linked circular list.
  External *prev_;
  External *next_;

 private:
  DISALLOW_COPY_AND_ASSIGN(External);
};

// Memory usage statistics.
struct MemoryUsage {
  // Number of bytes used for objects in heaps.
  int64 used_heap_bytes() const {
    return total_heap_size - unused_heap_bytes;
  }

  // Total memory allocated.
  int64 memory_allocated() const {
    return total_heap_size + num_handles * sizeof(Datum *);
  }

  // Total memory used.
  int64 memory_used() const {
    return used_heap_bytes() + used_handles() * sizeof(Datum *);
  }

  // Number of handles used.
  int used_handles() const {
    return num_handles - num_unused_handles -
           num_free_handles - num_dead_handles;
  }

  // Total number of symbols.
  int num_symbols() const {
    return num_bound_symbols + num_unbound_symbols + num_proxy_symbols;
  }

  int64 total_heap_size;    // total number of bytes allocates in heaps
  int64 unused_heap_bytes;  // number of unused bytes in heaps
  int num_heaps;            // number of heaps in store

  int num_handles;          // number of handles in handle table
  int num_unused_handles;   // number of unused handles
  int num_free_handles;     // number of free handles
  int num_dead_handles;     // number of dead handles

  int num_bound_symbols;    // number of bound symbols
  int num_unbound_symbols;  // number of unbound symbols
  int num_proxy_symbols;    // number of symbols bound to proxies
  int num_symbol_buckets;   // number of buckets in symbol hash table

  int num_gcs;              // number of garbage collections
  int64 gc_time;            // garbage collection time in microseconds
};

// The data for objects are stored in object heaps. An object heap is a
// contiguous memory area divided into two portions: used and unused. New
// objects are allocated from the unused portion until the heap is full. During
// garbage collection, the reachable objects in the heap are identified and the
// objects that are still alive are compacted into the beginning of the heap
// leaving a contiguous area at the end of the heap for allocating new objects.
class Heap : public Space<Datum> {
 public:
  Heap() : next_(nullptr), frozen_(false) {}

  // Next heap in store.
  Heap *next() const { return next_; }
  void set_next(Heap *next) { next_ = next; }

  // Read-only heap.
  bool frozen() const { return frozen_; }
  void set_frozen(bool frozen) { frozen_ = frozen; }

 private:
  // Next heap for store. All the heaps for a store are linked together in a
  // linked list.
  Heap *next_;

  // A heap can be frozen making the objects in the heap read-only.
  bool frozen_;

  DISALLOW_COPY_AND_ASSIGN(Heap);
};

// An object store maintains heaps of objects that are automatically reclaimed
// when they are no longer used. An object store can either be global or local.
// Objects can be added in a local store, whereas a global store is read-only.
// A global store can be accessed concurrently from multiple threads, but a
// local store is not thread-safe and should only be accessed from one thread at
// a time.
class Store {
 public:
  // Configuration options for store.
  struct Options {
    Options() {
      initial_heap_size = 32 * 1024;
      maximum_heap_size = 128 * (1 << 20);
      initial_handles = 1024;
      map_buckets = 1024;
      string_buckets = 1 << 20;
      expansion_free_fraction = 20;
      symbol_rebinding = false;
      local = this;
    }

    // Initial heap block size in bytes.
    int initial_heap_size;

    // Maximum heap block size in bytes.
    int maximum_heap_size;

    // Initial number of handles.
    int initial_handles;

    // Initial number of bucket in symbol hash table.
    int map_buckets;

    // Number of buckets for coalescing strings.
    int string_buckets;

    // Minimum fraction of free memory after GC to skip expansion.
    int expansion_free_fraction;

    // Allow symbols to be bound.
    bool symbol_rebinding;

    // Options for local store.
    Options *local;
  };

  // Initializes store with default configuration options.
  Store();

  // Initializes global store with custom configuration options.
  explicit Store(const Options *options);

  // Initializes local store.
  explicit Store(const Store *globals);

  // Deletes all objects in the store.
  ~Store();

  // Looks up symbol. A new unbound symbol is created if the symbol does not
  // already exist.
  Handle Symbol(Text name);
  Handle Symbol(Handle name);

  // Looks up symbol. Returns nil if the symbol does not exist.
  Handle ExistingSymbol(Text name) const;
  Handle ExistingSymbol(Handle name) const;

  // Looks up symbol and returns its value. A new symbol is created if the
  // symbol does not already exist. Also, if the symbol is not already bound, a
  // proxy is created.
  Handle Lookup(Text name);
  Handle Lookup(Handle name);

  // Looks up symbol and returns its value. Returns nil if the symbol does not
  // exist or it is not bound.
  Handle LookupExisting(Text name) const;
  Handle LookupExisting(Handle name) const;

  // Sets value for slot in  frame. If the frame has an existing slot with this
  // name, its value is updated. Otherwise a new slot is added to the frame. It
  // is not possible to update id slots of a frame with this method. If there
  // are multiple slots with this name, only the first one is updated. If you
  // want to update multiple slots it is faster to use a Builder object.
  void Set(Handle frame, Handle name, Handle value);

  // Adds new slot to frame. It is not possible to add id slots to a frame with
  // this method. If you want to add multiple slots it is faster to use a
  // Builder object.
  void Add(Handle frame, Handle name, Handle value);

  // Clones an existing frame. This can only be used on anonymous frames.
  // Returns handle to the new frame.
  Handle Clone(Handle frame);

  // Clones an existing frame and adds an additional slot to the clone. This can
  // only be used on anonymous frames. Returns handle to the new frame.
  Handle Extend(Handle frame, Handle name, Handle value);

  // Deletes all slots in a frame with a particular name.
  void Delete(Handle frame, Handle name);

  // Compares two objects by value. Anonymous frames can be compared by
  // reference.
  bool Equal(Handle x, Handle y, bool byref = false) const;

  // Computes a fingerprint for an object. This fingerprint is independent of
  // the specific handle values in this store. Fingerprints of frames with ids
  // only depend on the name, not the content of the frame. The object cannot
  // contain cycles. Anonymous frames can either be compared by value (default)
  // or by reference. If they are compared by reference, the fingerprint will
  // become dependent on the store.
  uint64 Fingerprint(Handle handle, bool byref = false, uint64 seed = 0) const;
  uint64 Fingerprint(ArrayDatum *array, int begin, int end, int step) const;
  uint64 Fingerprint(const Handle *begin, const Handle *end) const;

  // Returns a display name for the handle. This should only be used for display
  // purposes and should not be used as an alternative identifier for the
  // handle. If you want to get the text representation of an object you should
  // use the ToText function or a Printer object.
  string DebugString(Handle handle) const;

  // Dereferences handle and returns object in store.
  Datum *GetObject(Handle h) { return Deref(h); }
  const Datum *GetObject(Handle h) const { return Deref(h); }

  StringDatum *GetString(Handle h) { return Deref(h)->AsString(); }
  const StringDatum *GetString(Handle h) const { return Deref(h)->AsString(); }

  FrameDatum *GetFrame(Handle h) { return Deref(h)->AsFrame(); }
  const FrameDatum *GetFrame(Handle h) const { return Deref(h)->AsFrame(); }

  ArrayDatum *GetArray(Handle h) { return Deref(h)->AsArray(); }
  const ArrayDatum *GetArray(Handle h) const { return Deref(h)->AsArray(); }

  MapDatum *GetMap(Handle h) { return Deref(h)->AsMap(); }
  const MapDatum *GetMap(Handle h) const { return Deref(h)->AsMap(); }

  SymbolDatum *GetSymbol(Handle h) { return Deref(h)->AsSymbol(); }
  const SymbolDatum *GetSymbol(Handle h) const { return Deref(h)->AsSymbol(); }

  ProxyDatum *GetProxy(Handle h) { return Deref(h)->AsProxy(); }
  const ProxyDatum *GetProxy(Handle h) const { return Deref(h)->AsProxy(); }

  // Check if object is public, i.e. a frame with an id.
  bool IsPublic(Handle handle) const;
  bool IsAnonymous(Handle handle) const { return !IsPublic(handle); }

  // Resolve handle by following is: chain.
  Handle Resolve(Handle handle) const;

  // Get frame id.
  Text FrameId(Handle handle) const;

  // Freezes the store. This will convert all handles to global handles and make
  // the store read-only.
  void Freeze();

  // Merges occurrences of the same string. This saves memory by only keeping
  // one copy of each string value. This uses hashing, so it is not guaranteed
  // to find all identical strings.
  void CoalesceStrings();

  // Computes memory usage for store.
  void GetMemoryUsage(MemoryUsage *usage, bool quick = false) const;

  // Returns true if the store has been frozen.
  bool frozen() const { return frozen_; }

  // Global store for this store, or null if this is a global store.
  const Store *globals() const { return globals_; }

  // Returns handle for symbol table.
  Handle symbols() const { return symbols_; }

  // Returns the number of symbols in the symbol table.
  int num_symbols() const { return num_symbols_; }

  // Iterate all objects in the symbol table. This requires the store to be
  // stable during iteration to avoid invalidating the iterator.
  void ForAll(std::function<void(Handle handle)> callback) {
    const MapDatum *map = GetMap(symbols());
    for (Handle *bucket = map->begin(); bucket < map->end(); ++bucket) {
      Handle h = *bucket;
      while (!h.IsNil()) {
        const SymbolDatum *symbol = GetSymbol(h);
        if (symbol->bound()) callback(symbol->value);
        h = symbol->next;
      }
    }
  }

  // Checks if this handle is owned by this store.
  bool Owned(Handle handle) const {
    return handle.tag() == store_tag_;
  }

  // Returns the root list for the store.
  const Root *roots() const { return &roots_; }

  // Check if store is shared.
  bool shared() const { return refs_ != -1; }

  // Make store shared. This will set the reference count to one and the store
  // will be deleted when the reference count reaches zero.
  void Share();

  // Add reference count to store.
  void AddRef() const;

  // Release reference count on store. This will delete the store if this is the
  // last reference.
  void Release() const;

 public:
  // A reference in the handle table can be accessed as a heap object pointer or
  // as a pointer to the next element in the handle free list.
  union Reference {
    Datum *object;    // reference to heap object for handle
    Reference *next;  // reference to next element in handle free list
    uint64 bits;      // ensure that reference elements are 8 bytes
  };

  // The methods below are low-level methods for internal use.

  // Allocates uninitialized string object.
  Handle AllocateString(Word size);

  // Allocates and initializes string object.
  Handle AllocateString(Text str);

  // Allocates frame optionally replacing an existing frame.
  Handle AllocateFrame(Slot *begin, Slot *end, Handle original);
  Handle AllocateFrame(Slot *begin, Slot *end) {
    return AllocateFrame(begin, end, Handle::nil());
  }

  // Allocates empty frame.
  Handle AllocateFrame(Word slots);

  // Updates all the slots in the frame.
  void UpdateFrame(Handle handle, Slot *begin, Slot *end);

  // Allocates uninitialized array.
  Handle AllocateArray(Word length);

  // Allocates array and initializes its contents.
  Handle AllocateArray(const Handle *begin, const Handle *end);

  // Dereferences a handle and returns a pointer to the object data.
  Datum *Deref(Handle handle) {
    DCHECK(IsValidReference(handle));
    return pools_[handle.pool()][handle.idx()].object;
  }
  const Datum *Deref(Handle handle) const {
    DCHECK(IsValidReference(handle));
    return pools_[handle.pool()][handle.idx()].object;
  }

  // Checks basic type of object. This will return false for number types.
  bool IsType(Handle handle, Type type) const {
    if (!handle.IsRef() || handle.IsNil()) return false;
    return Deref(handle)->type() == type;
  }

  // Checks if handle refers to a proxy.
  bool IsProxy(Handle handle) const {
    return handle.IsRef() && !handle.IsNil() && Deref(handle)->IsProxy();
  }

  // Checks if handle refers to a symbol.
  bool IsSymbol(Handle handle) const {
    return handle.IsRef() && !handle.IsNil() && Deref(handle)->IsSymbol();
  }

  // Checks if handle refers to a frame.
  bool IsFrame(Handle handle) const {
    return handle.IsRef() && !handle.IsNil() && Deref(handle)->IsFrame();
  }

  // Checks if handle refers to a string.
  bool IsString(Handle handle) const {
    return handle.IsRef() && !handle.IsNil() && Deref(handle)->IsString();
  }

  // Returns nil if object type does not match.
  Handle Cast(Handle handle, Type type) const {
    return IsType(handle, type) ? handle : Handle::nil();
  }

  // Makes a local symbol if symbol is not owned. Otherwise this just returns
  // the symbol itself.
  SymbolDatum *LocalSymbol(SymbolDatum *symbol);

  // Replaces proxy with a frame.
  void ReplaceProxy(ProxyDatum *proxy, FrameDatum *frame);

  // Registers external objects.
  void RegisterExternal(External *external) {
    if (frozen_) {
      external->prev_ = external->next_ = external;
    } else {
      external->prev_ = &externals_;
      external->next_ = externals_.next_;
      externals_.next_->prev_ = external;
      externals_.next_ = external;
    }
  }

  // Adds and removes GC locks.
  void LockGC() { ++gc_locks_; }
  void UnlockGC() { if (--gc_locks_ == 0 && gc_pending_) GC(); }

  // Performs garbage collection.
  void GC();

  // Checks if store is pristine, i.e. the store only contains the standard
  // frames. This can be used for checking if a snapshot can be used for
  // restoring the store without overwriting any existing content.
  bool Pristine() const;

  // Resize symbol table.
  void ResizeSymbolTable();

  // Returns heap containing symbol table. This will return null if the heap
  // contains any objects besides the symbol table.
  Heap *GetSymbolHeap();

  // Allocates separate heap for symbol table.
  void AllocateSymbolHeap();

  // Iterator for enumerating all objects in the heaps. This will also iterate
  // over invalidated object in the heaps. The iterator will be invalidated by
  // any GCs. Please use this with care. This is primarily intended for
  // collecting statistics about an object store, and it should not be needed
  // under normal circumstances.
  class Iterator {
   public:
    explicit Iterator(const Store *store) {
      heap_ = store->first_heap_;
      current_ = heap_->base();
      end_ = heap_->end();
    }

    const Datum *next() {
      while (current_ == end_) {
        heap_ = heap_->next();
        if (heap_ == nullptr) return nullptr;
        current_ = heap_->base();
        end_ = heap_->end();
      }
      const Datum *object = current_;
      current_ = object->next();
      return object;
    }

   private:
    const Heap *heap_;
    const Datum *current_;
    const Datum *end_;
  };

 private:
  // Allocates heap object. The size is the payload size without the preamble.
  Datum *AllocateDatum(Type type, Word size) {
    // Determine the number of bytes needed for the object on the heap including
    // alignment. All objects must be aligned on the heap.
    Word bytes = Align(sizeof(Datum) + size);
    Datum *object;
    if (current_heap_->consume(bytes, &object)) {
      object->info = size | type;
      return object;
    } else {
      return AllocateDatumSlow(type, size);
    }
  }

  // Allocates memory for heap object when there is no more space in the current
  // heap.
  Datum *AllocateDatumSlow(Type type, Word size);

  // Allocates handle for object. A new handle is allocated from the free list.
  // If the free list is empty, the handle is allocated from the unsed part of
  // the handle table unless the handle table is full,
  Handle AllocateHandle(Datum *object) {
    Reference *ref;
    if (free_handle_ != nullptr) {
      ref = free_handle_;
      free_handle_ = ref->next;
    } else if (!handles_.consume(sizeof(Reference), &ref)) {
      return AllocateHandleSlow(object);
    }

    Handle handle = Handle::Ref(handles_.index(ref), store_tag_);
    ref->object = object;
    object->self = handle;
    return handle;
  }

  // Allocates handle when handle table is full.
  Handle AllocateHandleSlow(Datum *object);

  // Assigns heap object to handle.
  void Assign(Handle handle, Datum *object) {
    pools_[store_tag_][handle.idx()].object = object;
  }

  // Replaces heap object for a handle with a new object.
  void Replace(Handle handle, Datum *object) {
    // Mark old object as invalid.
    Deref(handle)->invalidate();

    // Update handle to point to new object.
    Assign(handle, object);

    // Update self handle in object.
    object->self = handle;
  }

  // Computes the hash value for a string and returns it as an integer handle.
  static Handle Hash(Text str);

  // Allocates proxy object for symbol.
  Handle AllocateProxy(Handle symbol);

  // Allocates symbol object using existing name string object.
  Handle AllocateSymbol(Handle name, Handle hash);

  // Allocates symbol object and name string object.
  Handle AllocateSymbol(Text name, Handle hash);

  // Looks up a symbol in symbol table. Returns handle for symbol or nil if
  // the symbol was not found. There is also a version where the hash value for
  // the name has been pre-computed.
  Handle FindSymbol(Text name) const;
  Handle FindSymbol(Text name, Handle hash) const;

  // Inserts symbol in symbol table.
  void InsertSymbol(SymbolDatum *symbol);

  // Checks if a handle is valid reference.
  bool IsValidReference(Handle handle) const;

  // Replaces all instances of a handle with another handle in all heap objects.
  // This is a very expensive operation that requires a complete heap traversal.
  void ReplaceHandle(Handle handle, Handle replacement);

  // Mark reachable objects.
  void Mark();

  // Compact heaps.
  void Compact();

  // Pointers to the global and local handle tables. These must be first in
  // the store object for fast dereferencing of object handles. These will be
  // pointers to the handle tables of the global and local stores.
  Reference *pools_[2];

  // Handle tag bits for objects in this store. This is Handle::kGlobalTag for
  // global stores and Handle::kLocalTag for local stores.
  Word store_tag_;

  // Reference to global store. This is null for global stores.
  const Store *globals_;

  // Stores must be frozen before being used as a global store for a local
  // store. When a store is frozen, it can no longer be changed.
  bool frozen_ = false;

  // Memory regions for storing object data. The heaps are linked together in
  // a linked list. The heaps are filled one by one until all the heaps are
  // full. Then the heaps needs to be garbage collected and if there is still
  // no room for new objects, a new heap will be allocated.
  Heap *current_heap_;
  Heap *first_heap_;
  Heap *last_heap_;

  // The handle table is used for storing references to objects. All access to
  // objects go through the handle table, which provides a level of indirection
  // that allows object to move dynamically, e.g. during garbage collection and
  // when symbols are resolved.
  static const size_t kMaxHandlesSize = Handle::kMaxHandles * sizeof(Reference);
  Reference *free_handle_;
  Space<Reference> handles_;

  // Root and external lists are used for locking objects that are referenced
  // externally.
  Root roots_;
  External externals_;

  // Symbol table.
  Handle symbols_;

  // Number of symbols in symbol table.
  int num_symbols_ = 0;

  // Number of hash buckets in the symbol table.
  int num_buckets_;

  // Reference count for shared stores. If the reference count is -1, the store
  // is not shared. Otherwise, the store is deleted when the reference count
  // goes to zero.
  mutable std::atomic<int> refs_{-1};

  // Number of GC locks. No garbage collection is performed as long as the
  // lock count is non-zero.
  int gc_locks_ = 0;
  bool gc_pending_ = false;

  // Number of garbage collections performed on store.
  int num_gcs_ = 0;

  // Time spent on garbage collection in microseconds.
  int64 gc_time_ = 0;

  // Number of dead handles after store has been frozen.
  int num_dead_handles_ = 0;

  // Configuration options for store.
  const Options *options_;

  // Default configuration options.
  static const Options kDefaultOptions;

  // Allow internal access for snapshots.
  friend class Snapshot;
};

// Utility class for GC locking in store.
class GCLock {
 public:
  // Lock GC in store.
  GCLock(Store *store) : store_(store) { store->LockGC(); }

  // Unlock GC in store.
  ~GCLock() { store_->UnlockGC(); }

 private:
  // Locked store.
  Store *store_;
};

// Adds root to store.
inline Root::Root(Store *store, Handle handle) {
  handle_ = handle;
  if (store == nullptr ||
      handle.IsNil() ||
      !handle.IsRef() ||
      !store->Owned(handle) ||
      store->frozen()) {
    next_ = prev_ = this;
  } else {
    Link(store->roots());
  }
}

inline void Root::InitRoot(Store *store, Handle handle) {
  handle_ = handle;
  if (store == nullptr ||
      handle.IsNil() ||
      !handle.IsRef() ||
      !store->Owned(handle) ||
      store->frozen()) {
    next_ = prev_ = this;
  } else {
    Link(store->roots());
  }
}

}  // namespace sling

#endif  // SLING_FRAME_STORE_H_

