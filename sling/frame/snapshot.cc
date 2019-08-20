// Copyright 2018 Google Inc.
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

#include "sling/frame/snapshot.h"

#include "sling/base/logging.h"
#include "sling/base/status.h"
#include "sling/base/types.h"
#include "sling/file/file.h"
#include "sling/frame/store.h"

namespace sling {

string Snapshot::Filename(const string &filename) {
  return filename + ".snap";
}

bool Snapshot::Valid(const string &filename) {
  // Get timestamp for store.
  FileStat stat;
  if (!File::Stat(filename, &stat)) return false;
  auto mtime = stat.mtime;

  // Try to open snapshot file.
  File *file;
  if (!File::Open(Filename(filename), "r", &file).ok()) return false;

  // Check that snapshot is not stale.
  bool ok = file->Stat(&stat).ok() && mtime < stat.mtime;

  // Check snapshot version.
  Header hdr;
  if (ok) ok = file->Read(&hdr, sizeof(Header)).ok();
  if (ok) ok = hdr.magic == MAGIC && hdr.version == VERSION;
  file->Close();
  return ok;
}

Status Snapshot::Read(Store *store, const string &filename) {
  // Only global stores can be restored from snapshot.
  if (store->globals() != nullptr) {
    return Status(1, "local store cannot be loaded from snapshot");
  }

  // Read snapshot header.
  File *file;
  Status st = File::Open(Filename(filename), "r", &file);
  if (!st.ok()) return st;

  Header hdr;
  st = file->Read(&hdr, sizeof(Header));
  if (!st.ok()) return st;

  if (hdr.magic != MAGIC) return Status(1, "invalid snapshot", filename);
  if (hdr.version != VERSION) return Status(1, "unsupported version", filename);

  // Delete existing heaps.
  Heap *heap = store->first_heap_;
  while (heap != nullptr) {
    Heap *next = heap->next();
    delete heap;
    heap = next;
  }
  store->first_heap_ = store->last_heap_ = store->current_heap_ = heap;

  // Read heaps from snapshot.
  Heap *symheap = nullptr;
  for (int i = 0; i < hdr.heaps; ++i) {
    // Read heap size.
    uint64 heapsize;
    st = file->Read(&heapsize, sizeof(uint64));
    if (!st.ok()) return st;

    // Allocate new heap.
    Heap *heap = new Heap();
    heap->reserve(heapsize);
    store->current_heap_ = heap;
    if (store->first_heap_ == nullptr) store->first_heap_ = heap;
    if (store->last_heap_ != nullptr) store->last_heap_->set_next(heap);
    store->last_heap_ = heap;

    // Read heap into memory.
    st = file->Read(heap->base(), heapsize);
    if (!st.ok()) return st;

    // Mark all space in heap as used.
    heap->set_end(heap->address(heapsize));

    // Check if this is the symbol table heap.
    if (hdr.symheap == i) symheap = heap;
  }

  // Allocate handle table.
  size_t handle_table_size = hdr.handles * sizeof(Store::Reference);
  auto &handles = store->handles_;
  handles.reserve(handle_table_size);
  handles.set_end(handles.base() + hdr.handles);
  store->pools_[Handle::kGlobal] = handles.base();

  // Clear handle table, leaving the nil entry intact.
  memset(handles.base() + 1, 0, (hdr.handles - 1) * sizeof(Store::Reference));

  // Restore handle table from self handles in objects. If snapshot has a
  // separate heap for the symbol table, all the other heaps are frozen.
  store->free_handle_ = nullptr;
  for (Heap *heap = store->first_heap_; heap != nullptr; heap = heap->next()) {
    bool freeze = (symheap != nullptr && heap != symheap);
    Datum *object = heap->base();
    Datum *end = heap->end();
    while (object < end) {
      if (!object->IsInvalid()) {
        // Update handle table from self handle.
        store->Assign(object->self, object);
        DCHECK(store->IsValidReference(object->self));

        // Set mark bit for objects in frozen heaps to prevent the GC from
        // traversing these objects.
        if (freeze) object->mark();
      }
      object = object->next();
    }
    if (freeze) heap->set_frozen(true);
  }

  // Set up symbol table.
  if (store->symbols_.bits != hdr.symtab) {
    return Status(1, "invalid symbol table handle", filename);
  }
  store->num_symbols_ = hdr.symbols;
  store->num_buckets_ = hdr.buckets;

  return file->Close();
}

Status Snapshot::Write(Store *store, const string &filename) {
  // Only global stores can be snapshot.
  if (store->globals() != nullptr) {
    return Status(1, "local store cannot be snapshot");
  }

  // Open output file.
  File *file;
  Status st = File::Open(Filename(filename), "w", &file);
  if (!st.ok()) return st;

  // Write header.
  Header hdr;
  hdr.magic = MAGIC;
  hdr.version = VERSION;
  hdr.handles = store->handles_.length();
  hdr.symtab = store->symbols_.bits;
  hdr.symbols = store->num_symbols_;
  hdr.buckets = store->num_buckets_;
  hdr.heaps = 0;
  hdr.symheap = -1;
  Heap *symheap = store->GetSymbolHeap();
  for (Heap *heap = store->first_heap_; heap != nullptr; heap = heap->next()) {
    if (heap == symheap) hdr.symheap = hdr.heaps;
    hdr.heaps++;
  }
  st = file->Write(&hdr, sizeof(Header));
  if (!st.ok()) {
    file->Close();
    return st;
  }

  // Write heaps.
  for (Heap *heap = store->first_heap_; heap != nullptr; heap = heap->next()) {
    uint64 heapsize = heap->size();
    st = file->Write(&heapsize, sizeof(uint64));
    if (st.ok()) st = file->Write(heap->base(), heapsize);
    if (!st) {
      file->Close();
      return st;
    }
  }

  return file->Close();
}

}  // namespace sling

