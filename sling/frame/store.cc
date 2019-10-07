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

#include "sling/frame/store.h"

#include <string>

#include "sling/base/clock.h"
#include "sling/base/logging.h"
#include "sling/string/strcat.h"
#include "sling/string/text.h"
#include "sling/util/city.h"

namespace sling {

// Initial heap with standard symbols.
// NB: This table depends on internal object layout, heap alignment, symbol
// hashing and pre-defined handle values. Please take this into consideration
// when making changes to this table.
static const Word kInitialHeap[] = {
  // id frame
  FRAME | PUBLIC |  8, 0x04, 0x04, 0x10,
  // isa frame
  FRAME | PUBLIC |  8, 0x08, 0x04, 0x14,
  // is frame
  FRAME | PUBLIC |  8, 0x0C, 0x04, 0x18,
  // id symbol
  SYMBOL         | 16, 0x10, 0x307A1C66, 0, 0x1C, 0x04,
  // isa symbol
  SYMBOL         | 16, 0x14, 0x6966506A, 0, 0x20, 0x08,
  // is symbol
  SYMBOL         | 16, 0x18, 0xC089FC02, 0, 0x24, 0x0C,
  // "id" string
  STRING         |  2, 0x1C, 'i' | ('d' << 8), 0,
  // "isa" string
  STRING         |  3, 0x20, 'i' | ('s' << 8) | ('a' << 16), 0,
  // "is" string
  STRING         |  2, 0x24, 'i' | ('s' << 8), 0,
};

// Size of pristine store.
static const int kPristineSymbols = 3;
static const int kPristineHandles = 11;

// Default store options.
const Store::Options Store::kDefaultOptions;

// Use city hash to compute hash values for strings.
static inline uint64 HashBytes(const void *ptr, size_t len) {
  return CityHash64(reinterpret_cast<const char *>(ptr), len);
}
static inline uint64 HashMix(const void *ptr, size_t len,
                             uint64 seed0, uint64 seed1) {
  return CityHash64WithSeeds(reinterpret_cast<const char *>(ptr), len,
                             seed0, seed1);
}
static inline uint64 HashMix(uint64 fp1, uint64 fp2) {
  return CityHash64Mix(fp1, fp2);
}

void Region::reserve(size_t bytes) {
  size_t used = size();
  DCHECK_LE(used, bytes);
  base_ = static_cast<Address>(realloc(base_, bytes));
  CHECK(base_ != nullptr || bytes == 0);
  CHECK_EQ((reinterpret_cast<uintptr_t>(base_) & (kObjectAlign - 1)), 0);
  end_ = base_ + used;
  limit_ = base_ + bytes;
  DCHECK(end_ <= limit_);
}

Address Region::alloc(size_t bytes) {
  if (limit_ - end_ < bytes) reserve(size() + bytes);
  Address ptr = end_;
  end_ += bytes;
  return ptr;
}

Address Region::expand(size_t bytes) {
  Address ptr = end_;
  Address next = ptr + bytes;
  if (next > limit_) {
    size_t size = limit_ - base_;
    size_t needed = end_ - base_ + bytes;
    if (size == 0) size = 1;
    while (size < needed) size *= 2;
    reserve(size);
    ptr = end_;
    next = ptr + bytes;
  }
  end_ = next;
  return ptr;
}

External::External() : prev_(this), next_(this) {}

External::External(Store *store) {
  store->RegisterExternal(this);
}

External::~External() {
  Unlink();
}

Store::Store() : Store(&kDefaultOptions) {}

Store::Store(const Options *options) : options_(options) {
  // Allocate initial heap.
  Heap *heap = new Heap();
  heap->reserve(options_->initial_heap_size);
  first_heap_ = last_heap_ = current_heap_ = heap;

  // The symbol table will be allocated later.
  symbols_ = Handle::nil();

  // Initialize handle table.
  handles_.reserve(options_->initial_handles);
  free_handle_ = nullptr;

  // Set up pools.
  globals_ = nullptr;
  store_tag_ = Handle::kGlobalTag;
  pools_[Handle::kGlobal] = handles_.base();
  pools_[Handle::kLocal] = nullptr;

  // Reserve the first global handle for the nil value. This is initialized
  // to point to an invalid memory location to trap any attempt at dereferencing
  // a nil handle.
  handles_.push()->bits = 0xdeadbeefdeadbeef;

  // Allocate standard heap objects.
  LockGC();
  Datum *begin = heap->add(sizeof(kInitialHeap) / sizeof(Datum));
  Datum *end = heap->end();
  memcpy(begin, kInitialHeap, sizeof(kInitialHeap));

  // Allocate handles for standard heap objects.
  for (Datum *datum = begin; datum < end; datum = datum->next()) {
    handles_.push()->object = datum;
  }

  // Allocate symbol table.
  num_buckets_ = options_->map_buckets;
  symbols_ = AllocateArray(num_buckets_);
  roots_.handle_ = symbols_;

  // Insert standard symbols into symbol table.
  for (Datum *datum = begin; datum < end; datum = datum->next()) {
    if (datum->IsSymbol()) InsertSymbol(datum->AsSymbol());
  }

  UnlockGC();
}

Store::Store(const Store *globals) : globals_(globals) {
  // Global store must be frozen.
  CHECK(globals->frozen_);

  // Add reference to shared global store.
  if (globals->shared()) globals->AddRef();

  // Get configuration options for local store.
  options_ = globals->options_->local;

  // Allocate initial heap.
  Heap *heap = new Heap();
  heap->reserve(options_->initial_heap_size);
  first_heap_ = last_heap_ = current_heap_ = heap;

  // Initialize handle table.
  handles_.reserve(options_->initial_handles);
  free_handle_ = nullptr;

  // Set up global and local pools.
  store_tag_ = Handle::kLocalTag;
  pools_[Handle::kGlobal] = globals_->pools_[Handle::kGlobal];
  pools_[Handle::kLocal] = handles_.base();

  // Allocate symbol map. The symbol table of a local store is initially only
  // a single bucket.
  num_buckets_ = 1;
  symbols_ = AllocateArray(num_buckets_);
  roots_.handle_ = symbols_;
}

Store::~Store() {
  // Make sure there are no references to store.
  CHECK(refs_ <= 0) << "Delete with live references to store";

  // Unlink roots and externals to prevent access to the store after it has been
  // destructed.
  roots_.Unlink();
  externals_.Unlink();

  // Delete all object heaps.
  Heap *heap = first_heap_;
  while (heap != nullptr) {
    Heap *next = heap->next();
    delete heap;
    heap = next;
  }

  // Release reference to shared global store.
  if (globals_ != nullptr && globals_->shared()) globals_->Release();
}

void Store::Share() {
  CHECK(!shared()) << "Store is already shared";
  refs_ = 1;
}

void Store::AddRef() const {
  int r = refs_.fetch_add(1);
  CHECK(r != -1) << "Store is not shared";
}

void Store::Release() const {
  int r = refs_.fetch_sub(1);
  if (r == 1) {
    delete this;
  } else {
    CHECK(r != -1) << "Store is not shared";
  }
}

Handle Store::AllocateString(Word size) {
  StringDatum *object = AllocateDatum(STRING, size)->AsString();
  return AllocateHandle(object);
}

Handle Store::AllocateString(Text str) {
  size_t size = str.size();
  StringDatum *object = AllocateDatum(STRING, size)->AsString();
  memcpy(object->data(), str.data(), size);
  return AllocateHandle(object);
}

Handle Store::AllocateFrame(Slot *begin, Slot *end, Handle original) {
  // Determine the handle for the new frame. The handle for the new frame can
  // be supplied as an argument, but if this is nil, we look at the id slots
  // for the new frame. If one of the id slots is set to a proxy (or a symbol
  // for a proxy), the handle for the proxy is used as the new handle for the
  // frame. The id slots will also be moved to the first slots in the frame.
  Handle handle = original;
  int ids = 0;
  Slot *next_id_slot = begin;
  for (Slot *s = begin; s < end; ++s) {
    // Check for id slots.
    if (s->name.IsId()) {
      // The id slot of a frame cannot be nil.
      CHECK(!s->value.IsNil());

      // Count the number of id slots for the new frame.
      ids++;

      // The id can either be a symbol or a proxy.
      Datum *value = Deref(s->value);
      Handle id = Handle::nil();
      if (value->IsSymbol()) {
        // Get local symbol. This will make a local symbol if it is not owned.
        SymbolDatum *symbol = LocalSymbol(value->AsSymbol());
        s->value = symbol->self;
        if (symbol->bound()) id = symbol->value;
      } else if (value->IsProxy()) {
        // Get symbol for proxy and make symbol local if it is not owned.
        ProxyDatum *proxy = value->AsProxy();
        SymbolDatum *symbol = LocalSymbol(GetSymbol(proxy->symbol));
        s->value = symbol->self;
        if (symbol->bound()) id = symbol->value;
      }

      // Only replace new frame handle if we don't already have a handle for it.
      if (handle.IsNil() && !id.IsNil()) handle = id;

      // Move slot to the next id slot if needed.
      if (s != next_id_slot) s->swap(next_id_slot);
      next_id_slot++;
    }
  }

  // Allocate frame object.
  size_t size = (end - begin) * sizeof(Slot);
  FrameDatum *frame = AllocateDatum(FRAME, size)->AsFrame();

  // Copy slots to new frame.
  Slot *t = frame->begin();
  for (Slot *s = begin; s < end; ++s, ++t) t->assign(s->name, s->value);

  // Allocate handle for frame unless it replaces a proxy.
  if (handle.IsNil()) {
    // Allocate new handle for frame.
    handle = AllocateHandle(frame);
  } else {
    // Make sure that the replaced handle is owned by this store.
    CHECK(Owned(handle));

    // Unbind existing object and replace it with the new frame.
    FrameDatum *existing = GetFrame(handle);
    CHECK(existing->IsFrame());
    for (Slot *slot = existing->begin(); slot < existing->end(); ++slot) {
      if (slot->name.IsId()) {
        // Unbind symbol from the existing frame.
        DCHECK(slot->value.IsRef());
        Datum *id = Deref(slot->value);
        DCHECK(id->IsSymbol());
        SymbolDatum *symbol = id->AsSymbol();
        DCHECK_EQ(symbol->value.raw(), handle.raw());
        symbol->value = symbol->self;
      }
    }
    Replace(handle, frame);
  }

  // If the frame has any id slots, bind these in the symbol table.
  if (ids > 0) {
    // Bind all id slots.
    for (Slot *slot = frame->begin(); ids > 0; ++slot) {
      if (!slot->name.IsId()) continue;
      ids--;
      CHECK(slot->value.IsRef());
      Datum *id = Deref(slot->value);
      if (id->IsSymbol()) {
        // Make sure the symbol is not already bound to another frame.
        SymbolDatum *symbol = id->AsSymbol();
        if (options_->symbol_rebinding) {
          CHECK(!symbol->marked()) << "no rebinding of frozen symbols";
        } else {
          CHECK(!symbol->bound()) << DebugString(symbol->self);
        }

        // Check that symbol is in the same pool as the frame.
        CHECK_EQ(handle.tag(), symbol->self.tag());

        // Bind symbol to frame.
        symbol->value = handle;
        frame->AddFlags(PUBLIC);
      } else if (id->IsProxy()) {
        // This proxy is not the one used for replacement, because otherwise the
        // proxy would have been replaced by the symbol above, so the frame has
        // multiple proxies. In this case we have to run through all frames and
        // replace the proxy handle with the handle of the new frame.
        CHECK_NE(slot->value.raw(), handle.raw());
        CHECK_EQ(handle.tag(), slot->value.tag());
        frame->AddFlags(id->typebits());
        ReplaceHandle(slot->value, handle);
        LOG(WARNING) << "double proxies are expensive";
      } else {
        LOG(FATAL) << "The value of an id slot must be a symbol or proxy";
      }
    }
  }

  return handle;
}

Handle Store::AllocateFrame(Word slots) {
  int size = slots * sizeof(Slot);
  FrameDatum *frame = AllocateDatum(FRAME, size)->AsFrame();
  memset(frame->payload(), 0, size);
  return AllocateHandle(frame);
}

void Store::UpdateFrame(Handle handle, Slot *begin, Slot *end) {
  // Make sure that handle is owned by this store.
  CHECK(Owned(handle));

  // Make sure that the frame has the right number of slots.
  FrameDatum *frame = GetFrame(handle);
  CHECK(frame->IsFrame());
  CHECK_EQ(end - begin, frame->end() - frame->begin());

  // Make sure the existing frame is anonymous.
  CHECK(frame->IsAnonymous());

  // Copy new slots to the frame.
  Slot *t = frame->begin();
  for (Slot *s = begin; s < end; ++s, ++t) {
    // Get slot name and value.
    Handle name = s->name;
    Handle value = s->value;

    // Save slot name and value in frame.
    t->name = name;
    t->value = value;

    // Bind frame to all symbols in the id slots.
    if (name.IsId()) {
      // The value of an id slot must be a symbol.
      DCHECK(!value.IsNil());
      DCHECK(value.IsRef());
      Datum *id = Deref(value);
      DCHECK(id->IsSymbol());
      SymbolDatum *symbol = id->AsSymbol();

      // Make sure the symbol is not already bound to another frame.
      if (options_->symbol_rebinding) {
        CHECK(!symbol->marked()) << "no rebinding of frozen symbols";
      } else {
        CHECK(!symbol->bound()) << DebugString(symbol->self);
      }

      // Check that symbol is in the same pool as the frame.
      CHECK_EQ(handle.tag(), symbol->self.tag());

      // Bind symbol to frame.
      symbol->value = handle;
      frame->AddFlags(PUBLIC);
    }
  }
}

Handle Store::AllocateArray(Word length) {
  // Allocate object.
  ArrayDatum *array = AllocateDatum(ARRAY, length * sizeof(Handle))->AsArray();

  // Initialize contents to nil values.
  for (Handle *e = array->begin(); e < array->end(); ++e) *e = Handle::nil();

  // Allocate handle.
  return AllocateHandle(array);
}

Handle Store::AllocateArray(const Handle *begin, const Handle *end) {
  // Allocate object.
  size_t size = (end - begin) * sizeof(Handle);
  ArrayDatum *array = AllocateDatum(ARRAY, size)->AsArray();

  // Initialize contents.
  memcpy(array->begin(), begin, size);

  // Allocate handle.
  return AllocateHandle(array);
}

void Store::Set(Handle frame, Handle name, Handle value) {
  // This method cannot be used for id slots because this would require updates
  // to the symbol table.
  CHECK(Owned(frame));
  CHECK(name != Handle::id());

  // Try to find slot with name.
  FrameDatum *datum = GetFrame(frame);
  CHECK(datum->IsFrame());
  for (Slot *s = datum->begin(); s < datum->end(); ++s) {
    if (s->name == name) {
      // Update slot and return.
      s->value = value;
      return;
    }
  }

  // Slot not found. Allocate a new replacement frame with an additional slot.
  int new_size = datum->size() + sizeof(Slot);
  FrameDatum *replacement = AllocateDatum(FRAME, new_size)->AsFrame();

  // Fetch frame again in case of GC.
  datum = GetFrame(frame);

  // Replace old frame.
  replacement->AddFlags(datum->typebits());
  Replace(frame, replacement);

  // Copy slots from original frame.
  Slot *t = replacement->begin();
  for (Slot *s = datum->begin(); s < datum->end(); ++s, ++t) {
    t->assign(s->name, s->value);
  }

  // Add new slot.
  t->assign(name, value);
}

void Store::Add(Handle frame, Handle name, Handle value) {
  // This method cannot be used for id slots because this would require updates
  // to the symbol table.
  CHECK(Owned(frame));
  CHECK(name != Handle::id());

  // Get existing frame.
  FrameDatum *datum = GetFrame(frame);
  CHECK(datum->IsFrame());

  // Allocate a new frame with an additional slot.
  int new_size = datum->size() + sizeof(Slot);
  FrameDatum *replacement = AllocateDatum(FRAME, new_size)->AsFrame();

  // Fetch frame again in case of GC.
  datum = GetFrame(frame);

  // Replace old frame.
  replacement->AddFlags(datum->typebits());
  Replace(frame, replacement);

  // Copy slots from original frame.
  Slot *t = replacement->begin();
  for (Slot *s = datum->begin(); s < datum->end(); ++s, ++t) {
    t->assign(s->name, s->value);
  }

  // Add new slot.
  t->assign(name, value);
}

Handle Store::Clone(Handle frame) {
  // Get existing frame.
  FrameDatum *datum = GetFrame(frame);
  CHECK(datum->IsFrame());

  // This method can only be used for anonymous frames.
  DCHECK(datum->IsAnonymous());

  // Allocate a clone frame.
  FrameDatum *clone = AllocateDatum(FRAME, datum->size())->AsFrame();

  // Fetch frame again in case of GC.
  datum = GetFrame(frame);

  // Copy slots from original frame.
  Slot *t = clone->begin();
  for (Slot *s = datum->begin(); s < datum->end(); ++s, ++t) {
    t->assign(s->name, s->value);
  }

  // Allocate and return handle for new frame.
  return AllocateHandle(clone);
}

Handle Store::Extend(Handle frame, Handle name, Handle value) {
  // Get existing frame.
  FrameDatum *datum = GetFrame(frame);
  CHECK(datum->IsFrame());

  // This method can only be used for anonymous frames.
  DCHECK(datum->IsAnonymous());
  DCHECK(name != Handle::id());

  // Allocate a clone frame with an additional slot.
  int new_size = datum->size() + sizeof(Slot);
  FrameDatum *clone = AllocateDatum(FRAME, new_size)->AsFrame();

  // Fetch frame again in case of GC.
  datum = GetFrame(frame);

  // Copy slots from original frame.
  Slot *t = clone->begin();
  for (Slot *s = datum->begin(); s < datum->end(); ++s, ++t) {
    t->assign(s->name, s->value);
  }

  // Add new slot.
  t->assign(name, value);

  // Allocate and return handle for new frame.
  return AllocateHandle(clone);
}

void Store::Delete(Handle frame, Handle name) {
  // Get frame.
  FrameDatum *datum = GetFrame(frame);
  CHECK(datum->IsFrame());

  // Id slots cannot be deleted.
  CHECK(name != Handle::id());

  // Delete all slots with name.
  Slot *slot = datum->begin();
  Slot *end = datum->end();
  while (slot < end && slot->name != name) slot++;
  if (slot == end) return;
  Slot *current = slot;
  while (slot < end) {
    if (slot->name == name) {
      slot++;
    } else {
      *current++ = *slot++;
    }
  }

  // Update frame size.
  Address limit = datum->limit();
  datum->resize((current - datum->begin()) * sizeof(Slot));

  // Create invalid object for the remainder.
  Datum *remainder = reinterpret_cast<Datum *>(current);
  remainder->invalidate();
  remainder->resize(limit - remainder->payload());
  remainder->self = Handle::nil();
}

Handle Store::Hash(Text str) {
  return Handle::Integer(HashBytes(str.data(), str.size()));
}

Handle Store::AllocateProxy(Handle symbol) {
  // Get symbol.
  SymbolDatum *sym = GetSymbol(symbol);
  CHECK(sym->IsSymbol());
  CHECK(!sym->bound());

  // Allocate proxy object.
  Datum *datum = AllocateDatum(FRAME, ProxyDatum::kSize);
  datum->info |= PROXY;
  ProxyDatum *proxy = datum->AsProxy();

  // Set id slot in proxy.
  proxy->id = Handle::id();
  proxy->symbol = symbol;
  proxy->AddFlags(PUBLIC);

  // Allocate handle for proxy.
  return AllocateHandle(proxy);
}

Handle Store::AllocateSymbol(Handle name, Handle hash) {
  // Allocate symbol object.
  SymbolDatum *symbol = AllocateDatum(SYMBOL, SymbolDatum::kSize)->AsSymbol();
  symbol->hash = hash;
  symbol->next = Handle::nil();
  symbol->name = name;
  Handle sym = AllocateHandle(symbol);
  symbol->value = sym;

  // Add symbol to symbol table.
  InsertSymbol(symbol);

  return sym;
}

Handle Store::AllocateSymbol(Text name, Handle hash) {
  // Allocate symbol initializing the name to nil.
  CHECK(!name.empty());
  Handle sym = AllocateSymbol(Handle::nil(), hash);

  // Allocate string object for symbol name.
  Handle str = AllocateString(name);
  GetSymbol(sym)->name = str;

  return sym;
}

void Store::InsertSymbol(SymbolDatum *symbol) {
  // Insert symbol in symbol table.
  GetMap(symbols_)->insert(symbol);
  num_symbols_++;

  // Resize symbol table if fill factor is more than 1:1, unless this would
  // make the symbol table exceed the maximum object size.
  if (num_symbols_ > num_buckets_ && num_buckets_ < kMapSizeLimit / 2) {
    ResizeSymbolTable();
  }
}

void Store::ResizeSymbolTable() {
  if (num_buckets_ == 1) {
    // Get the initial size from the options.
    num_buckets_ = options_->map_buckets;
  } else {
    // Double the number of buckets.
    num_buckets_ *= 2;
  }

  // Allocate new bucket array.
  int size = num_buckets_ * sizeof(Handle);
  MapDatum *map = AllocateDatum(ARRAY, size)->AsMap();
  for (Handle *h = map->begin(); h < map->end(); ++h) *h = Handle::nil();

  // Move all the symbols to the new symbol map.
  MapDatum *symbols = GetMap(symbols_);
  for (Handle *bucket = symbols->begin(); bucket < symbols->end(); ++bucket) {
    // Move all symbols in bucket to new map. If we encounter a symbol with
    // the mark bit set, it belongs to a frozen heap. These symbols need to be
    // inserted last in the bucket chain to ensusre that all non-frozen
    // symbols are traversed in the GC mark phase.
    Handle h = *bucket;
    while (!h.IsNil()) {
      SymbolDatum *symbol = GetSymbol(h);
      if (symbol->marked()) break;
      Handle next = symbol->next;
      map->insert(symbol);
      h = next;
    }

    // Handle buckets with frozen symbols.
    while (!h.IsNil()) {
      SymbolDatum *symbol = GetSymbol(h);
      Handle next = symbol->next;
      if (!symbol->marked()) {
        // Non-frozen symbols can be inserted at the head of the chain.
        map->insert(symbol);
      } else {
        // Insert frozen symbols after non-frozen symbols.
        Handle *b = map->bucket(symbol->hash);
        if (b->IsNil()) {
          // Empty bucket.
          *b = symbol->self;
          symbol->next = Handle::nil();
        } else {
          // If the head of the bucket chain is also frozen, the symbol can be
          // inserted at the head.
          SymbolDatum *head = GetSymbol(*b);
          if (head->marked()) {
            // Insert at head.
            symbol->next = *b;
            *b = symbol->self;
          } else {
            // Insert after non-frozen symbols.
            SymbolDatum *prev = head;
            while (!prev->marked() && !prev->next.IsNil()) {
              prev = GetSymbol(prev->next);
            }
            symbol->next = prev->next;
            prev->next = symbol->self;
          }
        }
      }

      // Next symbol in bucket chain.
      h = next;
    }
  }

  // Replace the old symbol table with the new one.
  Replace(symbols_, map);
}

Handle Store::FindSymbol(Text name, Handle hash) const {
  if (num_symbols_ > 0) {
    const MapDatum *symbols = GetMap(symbols_);
    Handle h = *symbols->bucket(hash);
    while (!h.IsNil()) {
      const SymbolDatum *symbol = GetSymbol(h);
      if (symbol->hash == hash) {
        const Datum *symname = GetObject(symbol->name);
        if (symname->IsString() && symname->AsString()->equals(name)) return h;
      }
      h = symbol->next;
    }
  }
  return Handle::nil();
}

Handle Store::FindSymbol(Text name) const {
  Handle hash = Hash(name);
  return FindSymbol(name, hash);
}

Handle Store::Symbol(Text name) {
  // Compute hash for name.
  Handle hash = Hash(name);

  // Try to look up symbol in local store.
  Handle h = FindSymbol(name, hash);
  if (!h.IsNil()) return h;

  // Try to look up symbol in global store.
  if (globals_ != nullptr) {
    h = globals_->FindSymbol(name, hash);
    if (!h.IsNil()) return h;
  }

  // Do not create new symbol if store is frozen.
  if (frozen_) return Handle::nil();

  // Symbol not found; create new symbol.
  return AllocateSymbol(name, hash);
}

Handle Store::Symbol(Handle name) {
  if (IsSymbol(name)) {
    // Name is a symbol, so just return it.
    return name;
  } else {
    // Get name string.
    Text str = GetString(name)->str();

    // Compute hash for name.
    Handle hash = Hash(str);

    // Try to look up symbol in local store.
    Handle h = FindSymbol(str, hash);
    if (!h.IsNil()) return h;

    // Try to look up symbol in global store.
    if (globals_ != nullptr) {
      h = globals_->FindSymbol(str, hash);
      if (!h.IsNil()) return h;
    }

    // Do not create new symbol if store is frozen.
    if (frozen_) return Handle::nil();

    // Symbol not found; create new symbol.
    return AllocateSymbol(name, hash);
  }
}

Handle Store::ExistingSymbol(Text name) const {
  // Compute hash for name.
  Handle hash = Hash(name);

  // Try to look up symbol in local store.
  Handle h = FindSymbol(name, hash);
  if (!h.IsNil()) return h;

  // Try to look up symbol in global store.
  if (globals_ != nullptr) {
    h = globals_->FindSymbol(name, hash);
    if (!h.IsNil()) return h;
  }

  return Handle::nil();
}

Handle Store::ExistingSymbol(Handle name) const {
  if (IsSymbol(name)) {
    // Name is a symbol, so just return it.
    return name;
  } else {
    // Get name string.
    Text str = GetString(name)->str();

    // Compute hash for name.
    Handle hash = Hash(str);

    // Try to look up symbol in local store.
    Handle h = FindSymbol(str, hash);
    if (!h.IsNil()) return h;

    // Try to look up symbol in global store.
    if (globals_ != nullptr) {
      h = globals_->FindSymbol(str, hash);
      if (!h.IsNil()) return h;
    }
  }

  return Handle::nil();
}

Handle Store::Lookup(Text name) {
  // Lookup or create symbol.
  Handle sym = Symbol(name);
  if (sym.IsNil()) return Handle::nil();

  // Return symbol value if it is already bound.
  SymbolDatum *symbol = GetSymbol(sym);
  if (symbol->bound()) return symbol->value;

  // If the symbol is unbound but the store is frozen then we can't create a
  // proxy, so just return nil.
  if (frozen_) return Handle::nil();

  // Symbol is unbound. Bind it to a new proxy.
  Handle proxy = AllocateProxy(sym);
  GetSymbol(sym)->value = proxy;
  return proxy;
}

Handle Store::Lookup(Handle name) {
  // Lookup or create symbol.
  Handle sym = Symbol(name);
  if (sym.IsNil()) return Handle::nil();

  // Return symbol value if it is already bound.
  SymbolDatum *symbol = GetSymbol(sym);
  if (symbol->bound()) return symbol->value;

  // If the symbol is unbound but the store is frozen then we can't create a
  // proxy, so just return nil.
  if (frozen_) return Handle::nil();

  // Symbol is unbound. Bind it to a new proxy.
  Handle proxy = AllocateProxy(sym);
  GetSymbol(sym)->value = proxy;
  return proxy;
}

Handle Store::LookupExisting(Text name) const {
  // Lookup symbol.
  Handle sym = ExistingSymbol(name);
  if (sym.IsNil()) return Handle::nil();

  // Return symbol value if it is already bound.
  const SymbolDatum *symbol = GetSymbol(sym);
  return symbol->bound() ? symbol->value : Handle::nil();
}

Handle Store::LookupExisting(Handle name) const {
  // Lookup symbol.
  Handle sym = ExistingSymbol(name);
  if (sym.IsNil()) return Handle::nil();

  // Return symbol value if it is already bound.
  const SymbolDatum *symbol = GetSymbol(sym);
  return symbol->bound() ? symbol->value : Handle::nil();
}

SymbolDatum *Store::LocalSymbol(SymbolDatum *symbol) {
  // Return symbol itself if it is owned.
  if (Owned(symbol->self)) return symbol;

  // Try to resolve symbol in the store.
  Text name = GetString(symbol->name)->str();
  Handle h = FindSymbol(name, symbol->hash);
  if (!h.IsNil()) return GetSymbol(h);

  // Create local symbol. The name string object from the global symbol is
  // reused in the local symbol.
  SymbolDatum *local = AllocateDatum(SYMBOL, SymbolDatum::kSize)->AsSymbol();
  local->hash = symbol->hash;
  local->next = Handle::nil();
  local->name = symbol->name;
  local->value = AllocateHandle(local);

  // Add symbol to local symbol table.
  InsertSymbol(local);

  return local;
}

bool Store::Equal(Handle x, Handle y, bool byref) const {
  // Trivial case.
  if (x == y) return true;

  if (x.IsNumber() || y.IsNumber()) {
    // Use the bit pattern for comparison.
    return x.bits == y.bits;
  } else {
    if (x.IsNil() || y.IsNil()) return false;
    const Datum *xdatum = GetObject(x);
    const Datum *ydatum = GetObject(y);
    if (xdatum->type() != ydatum->type()) return false;
    if (xdatum->size() != ydatum->size()) return false;
    switch (xdatum->type()) {
      case FRAME: {
        // Already tested if handles are equal.
        if (byref) return false;

        // Compare named frames by reference.
        const FrameDatum *xframe = xdatum->AsFrame();
        const FrameDatum *yframe = ydatum->AsFrame();
        if (xframe->IsPublic() || yframe->IsPublic()) return false;

        // Compare unnamed frames by value.
        const Slot *sx = xframe->begin();
        const Slot *sy = yframe->begin();
        while (sx < xframe->end()) {
          if (!Equal(sx->name, sy->name)) return false;
          if (!Equal(sx->value, sy->value)) return false;
          sx++;
          sy++;
        }
        return true;
      }
      case STRING: {
        // Compare string content.
        const StringDatum *xstr = xdatum->AsString();
        const StringDatum *ystr = ydatum->AsString();
        return xstr->equals(*ystr);
      }
      case SYMBOL: {
        // Already tested if handles are equal.
        return false;
      }
      case ARRAY: {
        // Compare all elements of the array.
        const ArrayDatum *xarray = xdatum->AsArray();
        const ArrayDatum *yarray = ydatum->AsArray();
        const Handle *hx = xarray->begin();
        const Handle *hy = yarray->begin();
        while (hx < xarray->end()) {
          if (!Equal(*hx++, *hy++)) return false;
        }
        return true;
      }
      default:
        return false;
    }
  }
}

// Fingerprint mixing constants.
enum FingerprintSeed : uint64 {
  FP_NUMBER =  0xd1ac3c3a168f9a23,
  FP_STRING =  0xedbf08a562d55ca0,
  FP_FRAME =   0x07a535307e971126,
  FP_SYMBOL =  0x06c498392bf66124,
  FP_ARRAY =   0x7e71d2f093c19cd1,
  FP_NIL =     0xe958f32bf433420c,
  FP_INVALID = 0x159ba7c32c364f9b,
  FP_HANDLE  = 0xd0bd1444ad3c9d01,
};

uint64 Store::Fingerprint(Handle handle, bool byref, uint64 seed) const {
  if (handle.IsNumber()) {
    // Use the bit pattern of the integer or float for hashing.
    return HashMix(HashMix(seed, FP_NUMBER), handle.bits);
  } else {
    if (handle.IsNil()) return HashMix(seed, FP_NIL);
    const Datum *datum = GetObject(handle);
    if (datum->IsFrame()) {
      const FrameDatum *frame = datum->AsFrame();
      if (frame->IsPublic()) {
        // Use the (first) frame id for hashing.
        Handle id = frame->get(Handle::id());
        DCHECK(!id.IsNil());
        const SymbolDatum *symbol = GetSymbol(id);
        return HashMix(seed, symbol->hash.bits);
      } else {
        uint64 fp = HashMix(seed, FP_FRAME);
        if (byref) {
          fp = HashMix(fp, handle.bits);
        } else {
          // Hash all slots in frame.
          for (const Slot *s = frame->begin(); s < frame->end(); ++s) {
            fp = Fingerprint(s->name, byref, fp);
            fp = Fingerprint(s->value, byref, fp);
          }
        }
        return fp;
      }
    } else {
      switch (datum->typebits()) {
        case STRING: {
          // Hash string content.
          const StringDatum *str = datum->AsString();
          return HashMix(str->data(), str->size(), seed, FP_STRING);
        }
        case SYMBOL: {
          // Use pre-computed symbol hash for fingerprint.
          const SymbolDatum *symbol = datum->AsSymbol();
          return HashMix(HashMix(seed, FP_SYMBOL), symbol->hash.bits);
        }
        case ARRAY: {
          // Hash all elements of the array.
          const ArrayDatum *array = datum->AsArray();
          uint64 fp = HashMix(seed, FP_ARRAY);
          for (const Handle *h = array->begin(); h < array->end(); ++h) {
            fp = Fingerprint(*h, byref, fp);
          }
          return fp;
        }
        default:
          return HashMix(seed, FP_INVALID);
      }
    }
  }
}

uint64 Store::Fingerprint(ArrayDatum *array,
                          int begin, int end, int step) const {
  // Hash array slice.
  uint64 fp = HashMix(0, FP_ARRAY);
  for (int i = begin; i != end; i += step) {
    Handle h = *(array->begin() + i);
    fp = Fingerprint(h, true, fp);
  }
  return fp;
}

uint64 Store::Fingerprint(const Handle *begin, const Handle *end) const {
  uint64 fp = HashMix(0, FP_ARRAY);
  for (const Handle *h = begin; h < end; ++h) {
    fp = Fingerprint(*h, true, fp);
  }
  return fp;
}

void Store::ReplaceProxy(ProxyDatum *proxy, FrameDatum *frame) {
  // Check that both the proxy and the frame are owned by the store.
  CHECK(Owned(proxy->self));
  CHECK(Owned(frame->self));

  // Swap the handles for the proxy and the frame.
  Assign(proxy->self, frame);
  Assign(frame->self, proxy);

  // Clear the proxy.
  proxy->id = Handle::nil();
  proxy->symbol = Handle::nil();
  proxy->info &= ~PROXY;

  // Update the self handles in the proxy and the frame.
  Handle tmp = proxy->self;
  proxy->self = frame->self;
  frame->self = tmp;
}

Datum *Store::AllocateDatumSlow(Type type, Word size) {
  // Object allocation not allowed in frozen store.
  CHECK(!frozen_);

  // Check for size overflow.
  Word bytes = Align(sizeof(Datum) + size);
  CHECK_LT(bytes, kObjectSizeLimit) << "Object too big";

  // This is called when the current heap is full.
  Datum *object;
  while (current_heap_->next() != nullptr) {
    // Switch to next heap.
    current_heap_ = current_heap_->next();

    // Try to allocate object on the new heap.
    if (current_heap_->consume(bytes, &object)) {
      object->info = size | type;
      return object;
    }
  }

  // Perform garbage collection.
  GC();

  // Compute the fraction of free memory after garbage collection. If this
  // fraction is too low, we allocate a new heap although there might be room
  // for the current allocation request in the current heaps. This is done to
  // prevent cascades of garbage collections when all the heaps are nearly full.
  int64 total = 0;
  int64 free = 0;
  for (Heap *heap = first_heap_; heap != nullptr; heap = heap->next()) {
    total += heap->capacity();
    free += heap->available();
  }

  if (free * options_->expansion_free_fraction > total) {
    // Retry allocation.
    current_heap_ = first_heap_;
    while (current_heap_ != nullptr) {
      // Try to allocate object on the new heap.
      if (current_heap_->consume(bytes, &object)) {
        object->info = size | type;
        return object;
      }

      // Switch to next heap.
      current_heap_ = current_heap_->next();
    }
  }

  // All heaps are still (nearly) full; compute size of new heap.
  size_t heap_size = last_heap_->capacity() * 2;
  if (heap_size > options_->maximum_heap_size) {
    heap_size = options_->maximum_heap_size;
  }
  while (heap_size < bytes) heap_size *= 2;

  // Allocate new heap.
  current_heap_ = new Heap();
  current_heap_->reserve(heap_size);
  last_heap_->set_next(current_heap_);
  last_heap_ = current_heap_;

  // Allocate object on new heap.
  CHECK(current_heap_->consume(bytes, &object));
  object->info = size | type;
  return object;
}

Handle Store::AllocateHandleSlow(Datum *object) {
  // Handle allocation not allowed in frozen store.
  CHECK(!frozen_);
  DCHECK(free_handle_ == nullptr);

  // Expand handle table.
  if (handles_.size() >= kMaxHandlesSize) LOG(FATAL) << "Handle overflow";
  size_t newsize = handles_.size() * 2;
  if (newsize > kMaxHandlesSize) newsize = kMaxHandlesSize;
  handles_.reserve(newsize);

  // Update the pool pointer to handle table.
  pools_[store_tag_] = handles_.base();

  // Allocate and initialize new handle.
  Reference *ref;
  CHECK(handles_.consume(sizeof(Reference), &ref));
  Handle handle = Handle::Ref(handles_.index(ref), store_tag_);
  ref->object = object;
  object->self = handle;

  return handle;
}

bool Store::IsPublic(Handle handle) const {
  if (!handle.IsRef()) return false;
  if (handle.IsNil()) return false;
  const Datum *datum = Deref(handle);
  if (!datum->IsFrame()) return false;
  return datum->AsFrame()->IsPublic();
}

Handle Store::Resolve(Handle handle) const {
  for (;;) {
    if (!handle.IsRef() || handle.IsNil()) return handle;
    const Datum *datum = Deref(handle);
    if (!datum->IsFrame()) return handle;

    const FrameDatum *frame = datum->AsFrame();
    if (frame->IsPublic()) return handle;

    Handle qua = frame->get(Handle::is());
    if (qua == Handle::nil()) return handle;
    handle = qua;
  }
}

Text Store::FrameId(Handle handle) const {
  if (!handle.IsRef() || handle.IsNil()) return Text();
  const Datum *datum = Deref(handle);
  if (!datum->IsFrame()) return Text();
  const FrameDatum *frame = datum->AsFrame();
  Handle id = frame->get(Handle::id());
  if (id.IsNil()) return Text();
  const Datum *iddatum = Deref(id);
  if (!iddatum->IsSymbol()) return Text();
  const SymbolDatum *symbol = iddatum->AsSymbol();
  const StringDatum *symstr = GetString(symbol->name);
  return symstr->str();
}

bool Store::Pristine() const {
  return globals_ == nullptr &&
         num_symbols_ == kPristineSymbols &&
         handles_.length() == kPristineHandles;
}

Heap *Store::GetSymbolHeap() {
  MapDatum *symbols = GetMap(symbols_);
  for (Heap *heap = first_heap_; heap != nullptr; heap = heap->next()) {
    if (heap->base() == symbols && heap->end() == symbols->next()) {
      return heap;
    }
  }
  return nullptr;
}

void Store::AllocateSymbolHeap() {
  // Check if symbol table is already in a separate heap.
  if (GetSymbolHeap() != nullptr) return;

  // Get current symbol table.
  Datum *symbols = Deref(symbols_);
  Word size = (symbols->next() - symbols) * sizeof(Datum);

  // Allocate new heap for symbol table.
  Heap *heap = new Heap();
  heap->reserve(size);
  last_heap_->set_next(heap);
  last_heap_ = heap;

  // Move symbol table to new heap.
  Datum *map;
  CHECK(heap->consume(size, &map));
  memcpy(map, symbols, size);

  // Replace the old symbol table with the new one.
  Replace(symbols_, map);
}

void Store::Mark() {
  // The marking stack keeps track of memory regions with handles that have not
  // yet been marked and traversed.
  Space<Range> stack;

  // Build table with all the roots.
  Space<Handle> root_table;
  const Root *root = &roots_;
  do {
    *root_table.push() = root->handle_;
    root = root->next_;
  } while (root != &roots_);

  // Add root table to the marking stack.
  Range *range = stack.push();
  range->begin = root_table.base();
  range->end = root_table.end();

  // Add all external object references to the marking stack.
  External *ext = &externals_;
  do {
    ext->GetReferences(stack.push());
    ext = ext->next_;
  } while (ext != &externals_);

  // Traverse all the objects reachable from the roots.
  Word pool_tag = store_tag_;
  Reference *pool = pools_[pool_tag];
  while (!stack.empty()) {
    Range *top = stack.top();
    if (top->empty()) {
      // Traversal of range has been completed.
      stack.pop();
    } else {
      // Get next handle in range.
      Handle h = *top->begin++;

      // Only owned objects need to be marked. Number handles (i.e. ints and
      // floats) represent themselves, and references to global objects in a
      // local store are regarded as static since the global store is frozen.
      if (!h.IsNil() && h.tag() == pool_tag) {
        // Dereference the handle. Here we take advantage of the fact that the
        // object is known to be owned so we can dereference the handle directly
        // through the owned handle table for the store.
        Datum *object = pool[h.idx()].object;

        // Mark the object if it is not already marked.
        if (!object->marked()) {
          object->mark();

          // Unless this is a binary object (i.e. string), we add the payload of
          // the object as a range that needs to be traversed and marked.
          if (!object->IsBinary()) object->range(stack.push());
        }
      }
    }
  }
}

void Store::Compact() {
  // The handles for the garbage collected objects are added to the handle
  // free list.
  Reference *fh = free_handle_;

  // Compact all the heaps.
  for (Heap *heap = first_heap_; heap != nullptr; heap = heap->next()) {
    // Do not compact frozen heaps.
    if (heap->frozen()) continue;

    // Traverse all the objects in the heap and move all the surviving objects
    // to the beginning of the heap.
    Datum *object = heap->base();
    Datum *end = heap->end();
    Datum *unused = object;
    while (object < end) {
      Datum *next = object->next();
      if (!object->IsInvalid()) {
        if (object->marked()) {
          // Object survived. Clear the mark.
          object->unmark();

          size_t size = Region::size(object, next);
          if (object != unused) {
            // Update handle table to point to the new object location.
            Assign(object->self, unused);

            // Move it to the new location at the start of the unused section.
            memmove(unused, object, size);
          }
          unused = Heap::address(unused, size);
        } else {
          // Object is dead. Free the associated handle.
          Reference *ref = handles_.base() + object->self.idx();
          ref->next = fh;
          fh = ref;
        }
      }
      object = next;
    }
    heap->set_end(unused);
  }

  // Start allocating from the first heap.
  current_heap_ = first_heap_;

  // Update the handle free list.
  free_handle_ = fh;
}

void Store::GC() {
  Clock timer;

  // Do not garbage collect a frozen store.
  if (frozen_) return;

  // Do not garbage collect a locked store, but indicate that GC is pending.
  if (gc_locks_ > 0) {
    gc_pending_ = true;
    return;
  }

  // Mark all the objects reachable from the roots.
  timer.start();
  Mark();
  timer.stop();
  int64 mark_time = timer.us();

  // Compact heaps.
  timer.start();
  Compact();
  gc_pending_ = false;
  timer.stop();
  int64 compact_time = timer.us();

  // Update statistics.
  int64 total_time = mark_time + compact_time;
  gc_time_ += total_time;
  num_gcs_++;

  VLOG(15) << "GC " << total_time << " us, "
           << "mark " << mark_time << " us, "
           << "compact " << compact_time << " us";
}

bool Store::IsValidReference(Handle handle) const {
  // Check that handle is a reference.
  if (handle.IsNil()) return true;
  if (!handle.IsRef()) {
    LOG(ERROR) << "Handle is not a reference";
    return false;
  }

  // Check that pool is allocated.
  const Space<Reference> *table;
  if (globals_ == nullptr) {
    // Global store.
    if (handle.IsGlobalRef()) {
      table = &handles_;
    } else {
      LOG(ERROR) << "Local handle in global store";
      return false;
    }
  } else {
    // Local store.
    if (handle.IsGlobalRef()) {
      table = &globals_->handles_;
    } else {
      table = &handles_;
    }
  }
  CHECK(table != nullptr);

  // Check bounds on handle table.
  Reference *ref = table->base() + handle.idx();
  if (ref < table->base() || ref >= table->end()) {
    LOG(ERROR) << "Handle outside handle table";
    return false;
  }

  // Check for valid pointer in handle table.
  if (ref->object == nullptr) {
    LOG(ERROR) << "Handle points to null object";
    return false;
  }

  // If the handle table entry contains a pointer into the handle table itself
  // it is part of the free list and therefore deleted.
  if (ref->next >= table->base() && ref->next < table->end()) {
    LOG(ERROR) << "Handle to reclaimed object";
    return false;
  }

  return true;
}

void Store::ReplaceHandle(Handle handle, Handle replacement) {
  // Scan the heaps and replace all instances of handle.
  for (Heap *heap = first_heap_; heap != nullptr; heap = heap->next()) {
    Datum *object = heap->base();
    Datum *end = heap->end();
    while (object < end) {
      if (!object->IsInvalid() && !object->IsBinary()) {
        Handle *begin = reinterpret_cast<Handle *>(object->payload());
        Handle *end = reinterpret_cast<Handle *>(object->limit());
        for (Handle *h = begin; h < end; ++h) {
          if (*h == handle) *h = replacement;
        }
      }
      object = object->next();
    }
  }

  // Replace handle in roots.
  const Root *root = &roots_;
  do {
    if (root->handle_ == handle) {
      const_cast<Root *>(root)->handle_ = replacement;
    }
    root = root->next_;
  } while (root != &roots_);

  // Replace handle in externals.
  External *ext = &externals_;
  do {
    Range range;
    ext->GetReferences(&range);
    for (Handle *h = range.begin; h < range.end; ++h) {
      if (*h == handle) *h = replacement;
    }
    ext = ext->next_;
  } while (ext != &externals_);
}

void Store::Freeze() {
  // Just return if store is already frozen.
  if (frozen_) return;

  // Local stores cannot be frozen.
  CHECK(globals_ == nullptr);

  // Run garbage collection to free up unused space.
  if (gc_locks_ == 0) GC();

  // Shrink all the heaps to fit the allocated data. This will force slow case
  // in object memory allocation where we check for frozen store.
  for (Heap *heap = first_heap_; heap != nullptr; heap = heap->next()) {
    // Shrink heap unless it is already full.
    if (!heap->full()) {
      // Resize heap.
      Datum *base = heap->base();
      heap->reserve(heap->size());
      if (heap->base() != base) {
        // Heap has moved so the handle table needs to be updated.
        Datum *object = heap->base();
        Datum *end = heap->end();
        while (object < end) {
          if (!object->IsInvalid()) Assign(object->self, object);
          object = object->next();
        }
      }
    }
  }

  // Clear all free handles.
  Reference *ref = free_handle_;
  while (ref != nullptr) {
    Reference *next = ref->next;
    ref->next = nullptr;
    ref = next;
    num_dead_handles_++;
  }
  free_handle_ = nullptr;

  // Shrink handle table to remove all the free handles at the end of the table.
  // This will force slow case on handle allocation where we check for frozen
  // store.
  while (!handles_.empty() && handles_.top() == nullptr) {
    handles_.pop();
    num_dead_handles_--;
  }
  handles_.reserve(handles_.size());
  pools_[store_tag_] = handles_.base();

  // Remove all roots from store. After the store has been frozen the roots no
  // longer need to be tracked.
  const Root *root = &roots_;
  do {
    const Root *next = root->next_;
    root->prev_ = root->next_ = root;
    root = next;
  } while (root != &roots_);

  External *ext = &externals_;
  do {
    External *next = ext->next_;
    ext->prev_ = ext->next_ = ext;
    ext = next;
  } while (ext != &externals_);

  // Store is now frozen.
  frozen_ = true;
}

void Store::CoalesceStrings() {
  // Do not coalesce strings in frozen store.
  if (frozen_) return;

  // Allocate cache for matching strings.
  Word num_buckets = options_->string_buckets;
  StringDatum **cache = new StringDatum *[num_buckets]();

  // Scan the heaps to find all strings.
  for (Heap *heap = first_heap_; heap != nullptr; heap = heap->next()) {
    Datum *object = heap->base();
    Datum *end = heap->end();
    while (object < end) {
      if (object->IsString()) {
        // Assign string to hash bucket in cache if it is not already used.
        StringDatum *str = object->AsString();
        Word b = HashBytes(str->data(), str->size()) % num_buckets;
        if (cache[b] == nullptr) cache[b] = str;
      }
      object = object->next();
    }
  }

  // Run through all objects and replace references to strings that match the
  // strings in the buckets.
  int num_replaced = 0;
  for (Heap *heap = first_heap_; heap != nullptr; heap = heap->next()) {
    Datum *object = heap->base();
    Datum *end = heap->end();
    while (object < end) {
      if (!object->IsInvalid() && !object->IsBinary()) {
        // Replace all string matches in the object values.
        Handle *begin = reinterpret_cast<Handle *>(object->payload());
        Handle *end = reinterpret_cast<Handle *>(object->limit());
        for (Handle *cell = begin; cell < end; ++cell) {
          // Check if value is a string.
          Handle h = *cell;
          if (h.IsNil() || !h.IsRef()) continue;
          Datum *o = Deref(h);
          if (!o->IsString()) continue;
          StringDatum *str = o->AsString();

          // Check if string is equal to the one in the cache. If it is the
          // string in the cache we leave it, but if it is a different string
          // with the same contents we replace it with the string in the cache.
          Word b = HashBytes(str->data(), str->size()) % num_buckets;
          StringDatum *intern = cache[b];
          DCHECK(intern != nullptr);
          if (intern == str) continue;
          if (str->equals(*intern)) {
            // Replace string with the cached string. The original string will
            // be removed during the next GC.
            *cell = intern->self;
            num_replaced++;
          }
        }
      }
      object = object->next();
    }
  }

  delete [] cache;
  VLOG(1) << num_replaced << " strings coalesced";
}

string Store::DebugString(Handle handle) const {
  if (handle.IsRef()) {
    if (handle.IsNil()) return "nil";
    const Datum *object = Deref(handle);

    // Get id for frame.
    if (object->IsFrame()) {
      const FrameDatum *frame = object->AsFrame();
      Handle id = frame->get(Handle::id());
      if (id.IsRef() && !id.IsNil()) object = Deref(id);
    }

    // Get name for symbol.
    if (object->IsSymbol()) {
      const SymbolDatum *symbol = object->AsSymbol();
      object = Deref(symbol->name);
    }

    // Return name for object.
    if (object->IsString()) {
      return StrCat(object->AsString()->str());
    } else if (object->IsArray()) {
      const ArrayDatum *array = object->AsArray();
      return StrCat("[" , array->length(), "@", handle.raw(), "]");
    } else {
      return StrCat("<<", handle.raw(), ">>");
    }
  } else if (handle.IsInt()) {
    return StrCat(handle.AsInt());
  } else if (handle.IsIndex()) {
    return StrCat("@", handle.AsIndex());
  } else if (handle.IsFloat()) {
    return StrCat(handle.AsFloat());
  } else {
    return StrCat("<<", handle.raw(), ">>");
  }
}

void Store::GetMemoryUsage(MemoryUsage *usage, bool quick) const {
  // Compute the number of bytes used by all heaps.
  usage->total_heap_size = 0;
  usage->unused_heap_bytes = 0;
  usage->num_heaps = 0;
  for (Heap *heap = first_heap_; heap != nullptr; heap = heap->next()) {
    usage->num_heaps++;
    usage->total_heap_size += heap->capacity();
    usage->unused_heap_bytes += heap->available();
  }

  // Compute handle table usage.
  usage->num_handles = handles_.capacity() / sizeof(Reference);
  usage->num_unused_handles = handles_.available() / sizeof(Reference);
  usage->num_dead_handles = num_dead_handles_;

  // Count the number of free elements in the handle table.
  int n = 0;
  if (!quick) {
    Reference *ref = free_handle_;
    while (ref != nullptr) {
      n++;
      ref = ref->next;
    }
  }
  usage->num_free_handles = n;

  // Symbol statistics.
  int bound = 0;
  int unbound = 0;
  int proxies = 0;
  if (quick) {
    bound = num_symbols_;
  } else {
    const MapDatum *symbols = GetMap(symbols_);
    for (Handle *bucket = symbols->begin(); bucket < symbols->end(); ++bucket) {
      Handle h = *bucket;
      while (!h.IsNil()) {
        const SymbolDatum *symbol = GetSymbol(h);
        if (symbol->unbound()) {
          unbound++;
        } else {
          const Datum *object = GetObject(symbol->value);
          if (object->IsProxy()) {
            proxies++;
          } else {
            bound++;
          }
        }
        h = symbol->next;
      }
    }
  }
  usage->num_bound_symbols = bound;
  usage->num_unbound_symbols = unbound;
  usage->num_proxy_symbols = proxies;
  usage->num_symbol_buckets = num_buckets_;

  // Garbage collection statistics.
  usage->num_gcs = num_gcs_;
  usage->gc_time = gc_time_;
}

}  // namespace sling

