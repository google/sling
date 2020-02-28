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

#ifndef SLING_FRAME_OBJECT_H_
#define SLING_FRAME_OBJECT_H_

#include <atomic>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sling/base/logging.h"
#include "sling/base/macros.h"
#include "sling/frame/store.h"
#include "sling/string/text.h"

namespace sling {

class String;
class Frame;
class Symbol;
class Array;

// Vector of handles that are tracked as external references.
class Handles : public std::vector<Handle>, public External {
 public:
  explicit Handles(Store *store) : External(store) {}

  void GetReferences(Range *range) override {
    range->begin = data();
    range->end = data() + size();
  }
};

// Vector of slots that are tracked as external references.
class Slots : public std::vector<Slot>, public External {
 public:
  explicit Slots(Store *store) : External(store) {}

  void GetReferences(Range *range) override {
    range->begin = reinterpret_cast<Handle *>(data());
    range->end = reinterpret_cast<Handle *>(data() + size());
  }
};

// Memory space for tracked handles.
class HandleSpace : public Space<Handle>, public External {
 public:
  explicit HandleSpace(Store *store) : External(store) {}

  void GetReferences(Range *range) override {
    range->begin = base();
    range->end = end();
  }
};

// Hash map and set keyed by handle.
template<typename T> using HandleMap =
  std::unordered_map<Handle, T, HandleHash>;
typedef std::unordered_set<Handle, HandleHash> HandleSet;

// Name with lazy lookup that can be initialized as static variables and
// then later be resolved using a Names object, e.g.:
//
// class MyClass {
//  public:
//   MyClass(Store *store) { names_.Bind(store); }
//   Handle foo(const Frame &f) { return f.Get(s_foo_); }
//   Handle bar(const Frame &f) { return f.Get(s_bar_); }
//  private:
//   Names names_;
//   Name s_foo_{names, "foo"};
//   Name s_bar_{names, "bar"};
// };

class Name;

class Names {
 public:
  // Adds name to name list.
  void Add(Name *name);

  // Resolves the names for all the name objects in the name list. Returns false
  // if some of the names could not be resolved. This is not an error, but it
  // means that the names need to be resolved with the Lookup() method.
  bool Bind(Store *store);
  bool Bind(const Store *store);

 private:
  Name *list_ = nullptr;
};

class Name {
 public:
  // Empty constructor.
  Name() {}

  // Initializes name without adding it to a name list.
  explicit Name(const string &name) : name_(name) {}

  // Initializes name and adds it to the names object.
  Name(Names &names, const string &name) : name_(name) { names.Add(this); }

  // Assign value from another name.
  void Assign(Name &other) {
    handle_ = other.handle_;
    store_ = other.store_;
  }

  // Looks up name, or use the handle if it has already been resolved.
  Handle Lookup(Store *store) const {
    if (!handle_.IsNil()) {
      DCHECK(store == store_ || store->globals() == store_);
      return handle_;
    } else {
      return store->Lookup(name_);
    }
  }

  // Accessors.
  Handle handle() const { return handle_; }
  void set_handle(Handle handle) { handle_ = handle; }
  const string &name() const { return name_; }
  void set_name(const string &name) { name_ = name; }
  const Store *store() const { return store_; }
  void set_store(const Store *store) { store_ = store; }

 private:
  // The Names class needs access to bind the name.
  friend class Names;

  // Handle for name. This is nil if the name has not been resolved.
  Handle handle_ = Handle::nil();

  // Symbol name.
  string name_;

  // Store for name.
  const Store *store_ = nullptr;

  // Next name in the name list.
  Name *next_ = nullptr;
};

class SharedNames : public Names {
 public:
  // Reference counting for shared names object.
  void AddRef() const { refs_.fetch_add(1); };
  void Release() const { if (refs_.fetch_sub(1) == 1) delete this; }

 protected:
  virtual ~SharedNames() { CHECK_EQ(refs_, 0); }

 private:
  // Reference count.
  mutable std::atomic<int> refs_{1};
};

// The Object class is the base class for holding references to heap objects.
// These can also contain tagged values for integer and floating-point numbers.
class Object : public Root {
 public:
  // Default constructor that initializes the object to the number zero.
  Object() : Root(nullptr, Handle::zero()), store_(nullptr) {}

  // Initializes object reference.
  Object(Store *store, Handle handle) : Root(store, handle), store_(store) {}

  // Looks up object in symbol table.
  Object(Store *store, Text id)
      : Root(store, store->Lookup(id)), store_(store) {}

  // Copy constructor.
  Object(const Object &other) : Root(other.handle_), store_(other.store_) {
    if (other.locked()) Link(&other);
  }

  // Assignment operator.
  Object &operator =(const Object &other);

  // Check if object is valid, i.e. is not nil.
  bool valid() const { return !IsNil(); }
  bool invalid() const { return IsNil(); }

  // Returns the object type. This can either be a simple type (integer or
  // float) or an complex type (string, frame, symbol, etc.).
  Type type() const;

  // Object type checking.
  bool IsInt() const { return handle_.IsInt(); }
  bool IsFloat() const { return handle_.IsFloat(); }
  bool IsNumber() const { return handle_.IsNumber(); }
  bool IsRef() const { return handle_.IsRef(); }
  bool IsGlobal() const { return handle_.IsGlobalRef(); }
  bool IsLocal() const { return handle_.IsLocalRef(); }

  // Value checking.
  bool IsNil() const { return handle_.IsNil(); }
  bool IsId() const { return handle_.IsId(); }
  bool IsFalse() const { return handle_.IsFalse(); }
  bool IsTrue() const { return handle_.IsTrue(); }
  bool IsZero() const { return handle_.IsZero(); }
  bool IsOne() const { return handle_.IsOne(); }

  // Returns object as integer.
  int AsInt() const { return handle_.AsInt(); }

  // Returns object as boolean.
  bool AsBool() const { return handle_.AsBool(); }

  // Returns object as floating-point number.
  float AsFloat() const { return handle_.AsFloat(); }

  // Type checking.
  bool IsString() const { return IsRef() && datum()->IsString(); }
  bool IsFrame() const { return IsRef() && datum()->IsFrame(); }
  bool IsSymbol() const { return IsRef() && datum()->IsSymbol(); }
  bool IsArray() const { return IsRef() && datum()->IsArray(); }

  // Converts to specific types. If the type does not match, nil is returned.
  String AsString() const;
  Frame AsFrame() const;
  Symbol AsSymbol() const;
  Array AsArray() const;

  // Returns a display name for the object.
  string DebugString() const { return store_->DebugString(handle_); }

  // Returns fingerprint for object.
  uint64 Fingerprint(uint64 seed = 0) const {
    return store_->Fingerprint(handle_, seed);
  }

  // Returns handle for value.
  Handle handle() const { return handle_; }

  // Returns reference to store for value.
  Store *store() const { return store_; }

 protected:
  // Dereferences object reference.
  Datum *datum() const { return store_->Deref(handle_); }

  // Object store for accessing datum.
  Store *store_;
};

// Reference to string in store.
class String : public Object {
 public:
  // Initializes to invalid string.
  String() : Object(nullptr, Handle::nil()) {}

  // Initializes a reference to an existing string object in the store.
  String(Store *store, Handle handle);

  // Creates new string in store.
  String(Store *store, Text str);

  // Copy constructor.
  String(const String &other) : Object(other) {}

  // Assignment operator.
  String &operator =(const String &other);

  // Returns the size of the string.
  int size() const { return str()->size(); }

  // Returns string contents of string object.
  string value() const {
    StringDatum *s = str();
    return string(s->data(), s->size());
  }

  // Returns string buffer.
  Text text() const { return str()->str(); }

  // Compares this string to a string buffer.
  bool equals(Text other) const {
    return str()->equals(other);
  }

 private:
  // Dereferences string reference.
  StringDatum *str() const { return datum()->AsString(); }
};

// Reference to symbol object in store.
class Symbol : public Object {
 public:
  // Initializes to invalid symbol.
  Symbol() : Object(nullptr, Handle::nil()) {}

  // Initializes a reference to an existing symbol object in the store.
  Symbol(Store *store, Handle handle);

  // Looks up symbol in symbol table.
  Symbol(Store *store, Text id);

  // Copy constructor which acquires a new lock for the symbol.
  Symbol(const Symbol &other) : Object(other) {}

  // Assignment operator.
  Symbol &operator =(const Symbol &other);

  // Returns symbol name.
  Object GetName() const;

  // Returns symbol value.
  Object GetValue() const;

  // Returns symbol name as text.
  Text name() const {
    StringDatum *symname = store()->GetString(symbol()->name);
    return Text(symname->data(), symname->size());
  }

  // Checks if symbol is bound.
  bool IsBound() const { return symbol()->bound(); }

 private:
  // Dereferences symbol reference.
  SymbolDatum *symbol() const { return datum()->AsSymbol(); }
};

// Reference to array object in store.
class Array : public Object {
 public:
  // Initializes to invalid array.
  Array() : Object(nullptr, Handle::nil()) {}

  // Initializes a reference to an existing array object in the store.
  Array(Store *store, Handle handle);

  // Copy constructor which acquires a new lock for the array.
  Array(const Array &other) : Object(other) {}

  // Creates a new array in the store.
  Array(Store *store, int size);
  Array(Store *store, const Handle *begin, const Handle *end);
  Array(Store *store, const std::vector<Handle> &contents)
      : Array(store, contents.data(), contents.data() + contents.size()) {}

  // Assignment operator.
  Array &operator =(const Array &other);

  // Returns the number of element in the array
  int length() const { return array()->length(); }

  // Gets element from array.
  Handle get(int index) const { return array()->get(index); }

  // Sets element in array.
  void set(int index, Handle value) const { *array()->at(index) = value; }

 private:
  // Dereferences array reference.
  ArrayDatum *array() const { return datum()->AsArray(); }
};

// Reference to frame in store.
class Frame : public Object {
 public:
  // Default constructor that initializes the object reference to nil.
  Frame() : Object(nullptr, Handle::nil()) {}

  // Initializes an reference to an existing frame in the store.
  Frame(Store *store, Handle handle);

  // Looks up frame in symbol table.
  Frame(Store *store, Text id);

  // Copy constructor which acquires a new lock for the frame reference.
  Frame(const Frame &other) : Object(other) {}

  // Creates a new frame in the store.
  Frame(Store *store, Slot *begin, Slot *end);
  Frame(Store *store, std::vector<Slot> *slots)
      : Frame(store, slots->data(), slots->data() + slots->size()) {}

  // Assignment operator.
  Frame &operator =(const Frame &other);

  // Checks if frame is a proxy.
  bool IsProxy() const { return frame()->IsProxy(); }

  // Checks if frame has an id.
  bool IsPublic() const { return frame()->IsPublic(); }

  // Checks if frame is anonymous, i.e. has no ids.
  bool IsAnonymous() const { return frame()->IsAnonymous(); }

  // Returns the number of slots in the frame.
  int size() const { return frame()->size() / sizeof(Slot); }

  // Returns slot name as handle.
  Handle name(int index) const {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, size());
    return frame()->begin()[index].name;
  }

  // Returns slot value as handle.
  Handle value(int index) const {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, size());
    return frame()->begin()[index].value;
  }

  // Gets (first) id for the object.
  Object id() const {
    // For proxies, the first slot is always the id slot.
    return IsProxy() ? Object(store_, value(0)) : Get(Handle::id());
  }

  // Returns the (first) id as a text.
  Text Id() const;

  // Checks if frame has named slot.
  bool Has(Handle name) const;
  bool Has(const Object &name) const;
  bool Has(const Name &name) const;
  bool Has(Text name) const;

  // Checks if frame has slot with value.
  bool Has(Handle name, Handle value) const;

  // Gets slot value.
  Object Get(Handle name) const;
  Object Get(const Object &name) const;
  Object Get(const Name &name) const;
  Object Get(Text name) const;

  // Gets slot value as frame.
  Frame GetFrame(Handle name) const;
  Frame GetFrame(const Object &name) const;
  Frame GetFrame(const Name &name) const;
  Frame GetFrame(Text name) const;

  // Gets slot value as symbol.
  Symbol GetSymbol(Handle name) const;
  Symbol GetSymbol(const Object &name) const;
  Symbol GetSymbol(const Name &name) const;
  Symbol GetSymbol(Text name) const;

  // Gets slot value as string.
  string GetString(Handle name) const;
  string GetString(const Object &name) const;
  string GetString(const Name &name) const;
  string GetString(Text name) const;

  // Gets slot value as text buffer.
  Text GetText(Handle name) const;
  Text GetText(const Object &name) const;
  Text GetText(const Name &name) const;
  Text GetText(Text name) const;

  // Get slot value as integer.
  int GetInt(Handle name, int defval) const;
  int GetInt(Handle name) const { return GetInt(name, 0); }
  int GetInt(const Object &name, int defval) const;
  int GetInt(const Object &name) const  { return GetInt(name, 0); }
  int GetInt(const Name &name, int defval) const;
  int GetInt(const Name &name) const  { return GetInt(name, 0); }
  int GetInt(Text name, int defval) const;
  int GetInt(Text name) const { return GetInt(name, 0); }

  // Get slot value as boolean.
  bool GetBool(Handle name, bool defval = false) const;
  bool GetBool(const Object &name, bool defval = false) const;
  bool GetBool(const Name &name, bool defval = false) const;
  bool GetBool(Text name, bool defval = false) const;

  // Get slot value as float.
  float GetFloat(Handle name) const;
  float GetFloat(const Object &name) const;
  float GetFloat(const Name &name) const;
  float GetFloat(Text name) const;

  // Get slot value as handle.
  Handle GetHandle(Handle name) const;
  Handle GetHandle(const Object &name) const;
  Handle GetHandle(const Name &name) const;
  Handle GetHandle(Text name) const;

  // Resolve slot value by following is: chain.
  Handle Resolve(Handle name) const;
  Handle Resolve(const Object &name) const;
  Handle Resolve(const Name &name) const;
  Handle Resolve(Text name) const;

  // Checks frame type, i.e. checks if frame has an isa/is slot with the type.
  bool IsA(Handle type) const;
  bool IsA(const Name &type) const;
  bool IsA(const Object &type) const;
  bool Is(Handle type) const;
  bool Is(const Name &type) const;
  bool Is(const Object &type) const;

  // Adds handle slot to frame.
  Frame &Add(Handle name, Handle value);
  Frame &Add(const Object &name, Handle value);
  Frame &Add(const Name &name, Handle value);
  Frame &Add(Text name, Handle value);
  Frame &Add(Handle value);

  // Adds slot to frame.
  Frame &Add(Handle name, const Object &value);
  Frame &Add(Handle name, const Name &value);
  Frame &Add(const Object &name, const Object &value);
  Frame &Add(const Object &name, const Name &value);
  Frame &Add(const Name &name, const Object &value);
  Frame &Add(const Name &name, const Name &value);
  Frame &Add(Text name, const Object &value);
  Frame &Add(Text name, const Name &value);

  // Adds integer slot to frame.
  Frame &Add(Handle name, int value);
  Frame &Add(const Object &name, int value);
  Frame &Add(const Name &name, int value);
  Frame &Add(Text name, int value);
  Frame &Add(int value);

  // Adds boolean slot to frame.
  Frame &Add(Handle name, bool value);
  Frame &Add(const Object &name, bool value);
  Frame &Add(const Name &name, bool value);
  Frame &Add(Text name, bool value);
  Frame &Add(bool value);

  // Adds floating point slot to frame.
  Frame &Add(Handle name, float value);
  Frame &Add(const Object &name, float value);
  Frame &Add(const Name &name, float value);
  Frame &Add(Text name, float value);
  Frame &Add(float value);

  Frame &Add(Handle name, double value);
  Frame &Add(const Object &name, double value);
  Frame &Add(const Name &name, double value);
  Frame &Add(Text name, double value);
  Frame &Add(double value);

  // Adds string slot to frame.
  Frame &Add(Handle name, Text value);
  Frame &Add(const Object &name, Text value);
  Frame &Add(const Name &name, Text value);
  Frame &Add(Text name, Text value);
  Frame &Add(Text value);

  Frame &Add(Handle name, const char *value);
  Frame &Add(const Object &name, const char *value);
  Frame &Add(const Name &name, const char *value);
  Frame &Add(Text name, const char *value);
  Frame &Add(const char *value);

  // Adds slot with symbol link to frame. This will link to a proxy if the
  // symbol is not already defined.
  Frame &AddLink(Handle name, Text symbol);
  Frame &AddLink(const Object &name, Text symbol);
  Frame &AddLink(const Name &name, Text symbol);
  Frame &AddLink(Text name, Text symbol);
  Frame &AddLink(Text symbol);

  // Adds isa: slot to frame.
  Frame &AddIsA(Handle type);
  Frame &AddIsA(const Object &type);
  Frame &AddIsA(const Name &type);
  Frame &AddIsA(Text type);
  Frame &AddIsA(const String &type);

  // Adds is: slot to frame.
  Frame &AddIs(Handle type);
  Frame &AddIs(const Object &type);
  Frame &AddIs(const Name &type);
  Frame &AddIs(Text type);
  Frame &AddIs(const String &type);

  // Sets slot to handle value.
  Frame &Set(Handle name, Handle value);
  Frame &Set(const Object &name, Handle value);
  Frame &Set(const Name &name, Handle value);
  Frame &Set(Text name, Handle value);

  // Sets slot to frame value.
  Frame &Set(Handle name, const Object &value);
  Frame &Set(Handle name, const Name &value);
  Frame &Set(const Object &name, const Object &value);
  Frame &Set(const Object &name, const Name &value);
  Frame &Set(const Name &name, const Object &value);
  Frame &Set(const Name &name, const Name &value);
  Frame &Set(Text name, const Object &value);
  Frame &Set(Text name, const Name &value);

  // Sets slot to integer value.
  Frame &Set(Handle name, int value);
  Frame &Set(const Object &name, int value);
  Frame &Set(const Name &name, int value);
  Frame &Set(Text name, int value);

  // Sets slot to boolean value.
  Frame &Set(Handle name, bool value);
  Frame &Set(const Object &name, bool value);
  Frame &Set(const Name &name, bool value);
  Frame &Set(Text name, bool value);

  // Sets slot to floating point value.
  Frame &Set(Handle name, float value);
  Frame &Set(const Object &name, float value);
  Frame &Set(const Name &name, float value);
  Frame &Set(Text name, float value);

  Frame &Set(Handle name, double value);
  Frame &Set(const Object &name, double value);
  Frame &Set(const Name &name, double value);
  Frame &Set(Text name, double value);

  // Sets slot to string value.
  Frame &Set(Handle name, Text value);
  Frame &Set(const Object &name, Text value);
  Frame &Set(const Name &name, Text value);
  Frame &Set(Text name, Text value);

  Frame &Set(Handle name, const char *value);
  Frame &Set(const Object &name, const char *value);
  Frame &Set(const Name &name, const char *value);
  Frame &Set(Text name, const char *value);

  // Sets slot to link to a frame. This will link to a proxy if the symbol is
  // not already defined.
  Frame &SetLink(Handle name, Text symbol);
  Frame &SetLink(const Object &name, Text symbol);
  Frame &SetLink(const Name &name, Text symbol);
  Frame &SetLink(Text name, Text symbol);

  // Iterator for iterating over all slots in a frame.
  class iterator {
   public:
    iterator(Store *store, const Slot *slot) : slot_(slot), store_(store) {
      store_->LockGC();
    }
    ~iterator() {
      store_->UnlockGC();
    }

    bool operator !=(const iterator &other) const {
      return slot_ != other.slot_;
    }

    const Slot &operator *() const { return *slot_; }

    const iterator &operator ++() { ++slot_; return *this; }

   private:
    const Slot *slot_;
    Store *store_;
  };

  const iterator begin() const { return iterator(store(), frame()->begin()); }
  const iterator end() const { return iterator(store(), frame()->end()); }

  // Slot predicate for iterator.
  typedef std::function<bool(const Slot *slot)> Predicate;

  // Iterator for iterating over all slots matching the predicate.
  class filterator {
   public:
    filterator(const Slot *slot, const Slot *end, const Predicate &predicate)
        : slot_(slot), end_(end), predicate_(predicate) {
      while (slot_ != end_ && !predicate_(slot_)) ++slot_;
    }

    bool operator !=(const filterator &other) const {
      return slot_ != other.slot_;
    }

    const Slot &operator *() const { return *slot_; }

    const filterator &operator ++() {
      ++slot_;
      while (slot_ != end_ && !predicate_(slot_)) ++slot_;
      return *this;
    }

   private:
    const Slot *slot_;
    const Slot *end_;
    const Predicate &predicate_;
  };

  // Filter for iterating over all slots in frame that match the predicate.
  class Filter {
   public:
    Filter(const Frame &frame, const Predicate &predicate)
        : frame_(frame), predicate_(predicate) {
      frame_.store()->LockGC();
    }

    ~Filter() { frame_.store()->UnlockGC(); }

    const filterator begin() const {
      return filterator(
          frame_.frame()->begin(),
          frame_.frame()->end(),
          predicate_);
    }

    const filterator end() const {
      return filterator(
          frame_.frame()->end(),
          frame_.frame()->end(),
          predicate_);
    }

   private:
    const Frame &frame_;
    Predicate predicate_;
  };

  // Iterates over all slots with a given name.
  Filter Slots(Handle name) const {
    return Filter(*this, [name](const Slot *slot) {
      return slot->name == name;
    });
  }

  Filter Slots(const Name &name) const {
    Handle h = name.Lookup(store());
    return Filter(*this, [h](const Slot *slot) {
      return slot->name == h;
    });
  }

  Filter Slots(Text name) const {
    Handle h = store()->Lookup(name);
    return Filter(*this, [h](const Slot *slot) {
      return slot->name == h;
    });
  }

  // Returns nil frame object.
  static Frame nil() { return Frame(); }

 private:
  // Dereferences frame reference.
  FrameDatum *frame() { return datum()->AsFrame(); }
  const FrameDatum *frame() const { return datum()->AsFrame(); }
};

// A builder is used for creating new frames in a store.
class Builder : public External {
 public:
  // Initializes object builder for store.
  explicit Builder(Store *store);

  // Initializes a builder from an existing frame and copies its slots.
  explicit Builder(const Frame &frame);
  Builder(Store *store, Handle handle);
  Builder(Store *store, Text id);

  ~Builder() override;

  // Adds handle slot to frame.
  Builder &Add(Handle name, Handle value);
  Builder &Add(const Object &name, Handle value);
  Builder &Add(const Name &name, Handle value);
  Builder &Add(Text name, Handle value);
  Builder &Add(Handle value);

  // Adds slot to frame.
  Builder &Add(Handle name, const Object &value);
  Builder &Add(Handle name, const Name &value);
  Builder &Add(const Object &name, const Object &value);
  Builder &Add(const Object &name, const Name &value);
  Builder &Add(const Name &name, const Object &value);
  Builder &Add(const Name &name, const Name &value);
  Builder &Add(Text name, const Object &value);
  Builder &Add(Text name, const Name &value);

  // Adds integer slot to frame.
  Builder &Add(Handle name, int value);
  Builder &Add(const Object &name, int value);
  Builder &Add(const Name &name, int value);
  Builder &Add(Text name, int value);
  Builder &Add(int value);

  // Adds boolean slot to frame.
  Builder &Add(Handle name, bool value);
  Builder &Add(const Object &name, bool value);
  Builder &Add(const Name &name, bool value);
  Builder &Add(Text name, bool value);
  Builder &Add(bool value);

  // Adds floating point slot to frame.
  Builder &Add(Handle name, float value);
  Builder &Add(const Object &name, float value);
  Builder &Add(const Name &name, float value);
  Builder &Add(Text name, float value);
  Builder &Add(float value);

  Builder &Add(Handle name, double value);
  Builder &Add(const Object &name, double value);
  Builder &Add(const Name &name, double value);
  Builder &Add(Text name, double value);
  Builder &Add(double value);

  // Adds string slot to frame.
  Builder &Add(Handle name, Text value);
  Builder &Add(const Object &name, Text value);
  Builder &Add(const Name &name, Text value);
  Builder &Add(Text name, Text value);
  Builder &Add(Text value);

  Builder &Add(Handle name, const char *value);
  Builder &Add(const Object &name, const char *value);
  Builder &Add(const Name &name, const char *value);
  Builder &Add(Text name, const char *value);
  Builder &Add(const char *value);

  // Adds array slot to frame.
  Builder &Add(Handle name, const Handles &value);
  Builder &Add(const Object &name, const Handles &value);
  Builder &Add(const Name &name, const Handles &value);
  Builder &Add(Text name, const Handles &value);

  // Adds slot with symbol link to frame. This will link to a proxy if the
  // symbol is not already defined.
  Builder &AddLink(Handle name, Text symbol);
  Builder &AddLink(const Object &name, Text symbol);
  Builder &AddLink(const Name &name, Text symbol);
  Builder &AddLink(Text name, Text symbol);
  Builder &AddLink(Text symbol);

  // Adds id: slot to frame.
  Builder &AddId(Handle id);
  Builder &AddId(const Object &id);
  Builder &AddId(Text id);
  Builder &AddId(const String &id);

  // Adds isa: slot to frame.
  Builder &AddIsA(Handle type);
  Builder &AddIsA(const Object &type);
  Builder &AddIsA(const Name &type);
  Builder &AddIsA(Text type);
  Builder &AddIsA(const String &type);

  // Adds is: slot to frame.
  Builder &AddIs(Handle type);
  Builder &AddIs(const Object &type);
  Builder &AddIs(const Name &type);
  Builder &AddIs(Text type);
  Builder &AddIs(const String &type);

  // Adds all the slots from another frame.
  Builder &AddFrom(Handle other);
  Builder &AddFrom(const Frame &other) { return AddFrom(other.handle()); }

  // Deletes slot(s) from frame.
  Builder &Delete(Handle name);
  Builder &Delete(const Object &name);
  Builder &Delete(const Name &name);
  Builder &Delete(Text name);

  // Sets slot to handle value.
  Builder &Set(Handle name, Handle value);
  Builder &Set(const Object &name, Handle value);
  Builder &Set(const Name &name, Handle value);
  Builder &Set(Text name, Handle value);

  // Sets slot to frame value.
  Builder &Set(Handle name, const Object &value);
  Builder &Set(Handle name, const Name &value);
  Builder &Set(const Object &name, const Object &value);
  Builder &Set(const Object &name, const Name &value);
  Builder &Set(const Name &name, const Object &value);
  Builder &Set(const Name &name, const Name &value);
  Builder &Set(Text name, const Object &value);
  Builder &Set(Text name, const Name &value);

  // Sets slot to integer value.
  Builder &Set(Handle name, int value);
  Builder &Set(const Object &name, int value);
  Builder &Set(const Name &name, int value);
  Builder &Set(Text name, int value);

  // Sets slot to boolean value.
  Builder &Set(Handle name, bool value);
  Builder &Set(const Object &name, bool value);
  Builder &Set(const Name &name, bool value);
  Builder &Set(Text name, bool value);

  // Sets slot to floating point value.
  Builder &Set(Handle name, float value);
  Builder &Set(const Object &name, float value);
  Builder &Set(const Name &name, float value);
  Builder &Set(Text name, float value);

  Builder &Set(Handle name, double value);
  Builder &Set(const Object &name, double value);
  Builder &Set(const Name &name, double value);
  Builder &Set(Text name, double value);

  // Sets slot to string value.
  Builder &Set(Handle name, Text value);
  Builder &Set(const Object &name, Text value);
  Builder &Set(const Name &name, Text value);
  Builder &Set(Text name, Text value);

  Builder &Set(Handle name, const char *value);
  Builder &Set(const Object &name, const char *value);
  Builder &Set(const Name &name, const char *value);
  Builder &Set(Text name, const char *value);

  // Sets slot to link to a frame. This will link to a proxy if the symbol is
  // not already defined.
  Builder &SetLink(Handle name, Text symbol);
  Builder &SetLink(const Object &name, Text symbol);
  Builder &SetLink(const Name &name, Text symbol);
  Builder &SetLink(Text name, Text symbol);

  // Creates frame from the slots in the frame builder.
  Frame Create();

  // Update existing frame with new slots.
  void Update();

  // Clears all the slots.
  Builder &Clear() { slots_.reset(); return *this; }

  // Checks if this is a new frame, i.e. the existing handle is nil or points
  // to a proxy.
  bool IsNew() const { return handle_.IsNil() || store_->IsProxy(handle_); }

  // Returns the range of object references. This is used by the GC to keep all
  // the referenced objects in the object builder slots alive.
  void GetReferences(Range *range) override;

  // Returns the store behind the builder.
  Store *store() { return store_; }

  // Returns the handle for the builder.
  Handle handle() { return handle_; }

 private:
  // Initial number of slots reserved.
  static const int kInitialSlots = 16;

  // Adds new slot to the frame builder.
  Slot *NewSlot() {
    Slot *slot = slots_.push();
    slot->name = Handle::nil();
    slot->value = Handle::nil();
    return slot;
  }

  // Finds first slot with name, or adds a new slot with that name.
  Slot *NamedSlot(Handle name) {
    for (Slot *slot = slots_.base(); slot < slots_.end(); ++slot) {
      if (slot->name == name) return slot;
    }
    Slot *slot = slots_.push();
    slot->name = name;
    slot->value = Handle::nil();
    return slot;
  }

  // Store where frames should be created.
  Store *store_;

  // Handle for frame to be replaced, or nil if we should construct a new
  // frame.
  Handle handle_;

  // Slots for frame builder.
  Space<Slot> slots_;

  DISALLOW_IMPLICIT_CONSTRUCTORS(Builder);
};

// Comparison operators.
inline bool operator ==(const Object &a, const Object &b) {
  return a.handle() == b.handle();
}

inline bool operator !=(const Object &a, const Object &b) {
  return a.handle() != b.handle();
}

inline bool operator ==(const Object &a, Handle b) {
  return a.handle() == b;
}

inline bool operator !=(const Object &a, Handle b) {
  return a.handle() != b;
}

inline bool operator ==(const Object &a, const Name &b) {
  return a.handle() == b.Lookup(a.store());
}

inline bool operator !=(const Object &a, const Name &b) {
  return a.handle() != b.Lookup(a.store());
}

inline bool operator ==(Handle a, const Object &b) {
  return a == b.handle();
}

inline bool operator !=(Handle a, const Object &b) {
  return a != b.handle();
}

inline bool operator ==(Handle a, const Name &b) {
  CHECK(!b.handle().IsNil()) << "Comparison with unresolved name";
  return a == b.handle();
}

inline bool operator !=(Handle a, const Name &b) {
  CHECK(!b.handle().IsNil()) << "Comparison with unresolved name";
  return a != b.handle();
}

inline bool operator ==(const Name &a, const Object &b) {
  return a.Lookup(b.store()) == b.handle();
}

inline bool operator !=(const Name &a, const Object &b) {
  return a.Lookup(b.store()) != b.handle();
}

inline bool operator ==(const Name &a, Handle b) {
  CHECK(!a.handle().IsNil()) << "Comparison with unresolved name";
  return a.handle() == b;
}

inline bool operator !=(const Name &a, Handle b) {
  CHECK(!a.handle().IsNil()) << "Comparison with unresolved name";
  return a.handle() != b;
}

inline bool operator ==(const Name &a, const Name &b) {
  CHECK(!a.handle().IsNil()) << "Comparison with unresolved name";
  CHECK(!b.handle().IsNil()) << "Comparison with unresolved name";
  return a.handle() == b.handle();
}

inline bool operator !=(const Name &a, const Name &b) {
  CHECK(!a.handle().IsNil()) << "Comparison with unresolved name";
  CHECK(!b.handle().IsNil()) << "Comparison with unresolved name";
  return a.handle() != b.handle();
}

}  // namespace sling

#endif  // SLING_FRAME_OBJECT_H_

