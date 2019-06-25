// Copyright (c) 1994-2006 Sun Microsystems Inc.
// All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// - Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// - Redistribution in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// - Neither the name of Sun Microsystems or the names of contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// The original source code covered by the above license above has been
// modified significantly by Google Inc.
// Copyright 2012 the V8 project authors. All rights reserved.
// Copyright 2017 Google Inc. All rights reserved.

#include <stdlib.h>
#include <sys/mman.h>

#include "third_party/jit/code.h"

#include "sling/base/logging.h"
#include "third_party/jit/types.h"

namespace sling {
namespace jit {

int Label::pos() const {
  if (pos_ < 0) return -pos_ - 1;
  if (pos_ > 0) return  pos_ - 1;
  LOG(FATAL) << "Unresolved label";
  return 0;
}

CodeGenerator::CodeGenerator(void *buffer, int buffer_size) {
  own_buffer_ = buffer == nullptr;
  if (buffer_size == 0) buffer_size = kMinimalBufferSize;
  DCHECK(buffer_size > 0);
  if (own_buffer_) buffer = malloc(buffer_size);
  buffer_ = static_cast<byte *>(buffer);
  buffer_size_ = buffer_size;

  pc_ = buffer_;
}

void CodeGenerator::GrowBuffer() {
  DCHECK(buffer_overflow());
  if (!own_buffer_) LOG(FATAL) << "external code buffer is too small";

  // Expand code buffer.
  byte *old_buffer = buffer_;
  buffer_size_ *= 2;
  buffer_ = static_cast<byte*>(realloc(buffer_, buffer_size_));
  intptr_t pc_delta = buffer_ - old_buffer;
  pc_ += pc_delta;

  // Relocate internal references.
  for (auto pos : refs_) {
    intptr_t *p = reinterpret_cast<intptr_t *>(buffer_ + pos);
    *p += pc_delta;
  }

  DCHECK(!buffer_overflow());
}

CodeGenerator::~CodeGenerator() {
  if (own_buffer_) free(buffer_);
}

void CodeGenerator::bind(Label *l) {
  bind_to(l, pc_offset());
}

void CodeGenerator::bind_to(Label *l, int pos) {
  DCHECK(!l->is_bound());  // label may only be bound once
  DCHECK(0 <= pos && pos <= pc_offset());  // position must be valid
  if (l->is_linked()) {
    int current = l->pos();
    int next = long_at(current);
    int sl = 0;
    if (next < 0) {
      sl = 1;
      next = -next;
    }
    while (next != current) {
      if (current >= 4 && long_at(current - 4) == 0) {
        // Absolute address.
        intptr_t imm64 = reinterpret_cast<intptr_t>(buffer_ + pos);
        *reinterpret_cast<intptr_t *>(addr_at(current - 4)) = imm64;
        refs_.push_back(current - 4);
      } else {
        // Relative address, relative to point after address.
        int imm32 = pos - (current + sizeof(int32_t) + sl);
        long_at_put(current, imm32);
      }
      current = next;
      next = long_at(next);
      sl = 0;
      if (next < 0) {
        sl = 1;
        next = -next;
      }
    }
    // Fix up last fixup on linked list.
    if (current >= 4 && long_at(current - 4) == 0) {
      // Absolute address.
      intptr_t imm64 = reinterpret_cast<intptr_t>(buffer_ + pos);
      *reinterpret_cast<intptr_t *>(addr_at(current - 4)) = imm64;
      refs_.push_back(current - 4);
    } else {
      // Relative address, relative to point after address.
      int imm32 = pos - (current + sizeof(int32_t) + sl);
      long_at_put(current, imm32);
    }
  }
  while (l->is_near_linked()) {
    int fixup_pos = l->near_link_pos();
    int offset_to_next =
        static_cast<int>(*reinterpret_cast<int8_t *>(addr_at(fixup_pos)));
    DCHECK(offset_to_next <= 0);
    int disp = pos - (fixup_pos + sizeof(int8_t));
    CHECK(is_int8(disp));
    set_byte_at(fixup_pos, disp);
    if (offset_to_next < 0) {
      l->link_to(fixup_pos + offset_to_next, Label::kNear);
    } else {
      l->UnuseNear();
    }
  }
  l->bind_to(pos);
}

void CodeGenerator::AddExtern(const string &symbol, Address address,
                              bool relative) {
  // Try to find existing external reference.
  int index = -1;
  for (int i = 0; i < externs_.size(); ++i) {
    if (address == externs_[i].address && symbol == externs_[i].symbol) {
      index = i;
      break;
    }
  }

  // Add new external symbol.
  if (index == -1) {
    index = externs_.size();
    externs_.emplace_back(symbol, address);
  }

  // Add reference to external symbol.
  externs_[index].refs.emplace_back(pc_offset(), relative);
}

Code::Code(void *code, int size) : memory_(nullptr), size_(0) {
  Allocate(code, size);
}

Code::~Code() {
  if (memory_ != nullptr) munmap(memory_, size_);
}

void Code::Allocate(void *code, int size) {
  // Allocate r/w memory.
  CHECK(memory_ == nullptr);
  memory_ = static_cast<byte *>(
                mmap(nullptr, size,
                     PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE,
                     0, 0));
  CHECK(memory_ != nullptr);
  size_ = size;

  // Copy code block to allocated memory.
  memcpy(memory_, code, size);

  // Make code executable and remove write permissions.
  int rc = mprotect(memory_, size_, PROT_READ | PROT_EXEC);
  CHECK_EQ(rc, 0);
}

}  // namespace jit
}  // namespace sling

