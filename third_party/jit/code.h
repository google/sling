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

#ifndef JIT_CODE_H_
#define JIT_CODE_H_

#include <deque>
#include <string>
#include <vector>

#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "third_party/jit/memory.h"

namespace sling {
namespace jit {

// Labels represent pc locations; they are typically jump or call targets.
// After declaration, a label can be freely used to denote known or (yet)
// unknown pc location. CodeGenerator::bind() is used to bind a label to the
// current pc. A label can be bound only once.
class Label {
 public:
  enum Distance {
    kNear, kFar
  };

  inline Label() {
    Unuse();
    UnuseNear();
  }

  inline ~Label() {
    DCHECK(!is_linked());
    DCHECK(!is_near_linked());
  }

  inline void Unuse() { pos_ = 0; }
  inline void UnuseNear() { near_link_pos_ = 0; }

  inline bool is_bound() const { return pos_ <  0; }
  inline bool is_unused() const { return pos_ == 0 && near_link_pos_ == 0; }
  inline bool is_linked() const { return pos_ >  0; }
  inline bool is_near_linked() const { return near_link_pos_ > 0; }

  // Returns the position of bound or linked labels. Cannot be used
  // for unused labels.
  int pos() const;
  int near_link_pos() const { return near_link_pos_ - 1; }

  void bind_to(int pos)  {
    pos_ = -pos - 1;
    DCHECK(is_bound());
  }

  void link_to(int pos, Distance distance = kFar) {
    if (distance == kNear) {
      near_link_pos_ = pos + 1;
      DCHECK(is_near_linked());
    } else {
      pos_ = pos + 1;
      DCHECK(is_linked());
    }
  }

 private:
  // pos_ encodes both the binding state (via its sign)
  // and the binding position (via its value) of a label.
  //
  // pos_ <  0  bound label, pos() returns the jump target position
  // pos_ == 0  unused label
  // pos_ >  0  linked label, pos() returns the last reference position
  int pos_;

  // Behaves like |pos_| in the "> 0" case, but for near jumps to this label.
  int near_link_pos_;

  friend class CodeGenerator;
};

// An external symbol is a reference to code or data outside the code buffer
// of the code generator.
struct Extern {
  struct Ref {
    Ref(int offset, bool relative) : offset(offset), relative(relative) {}
    int offset;     // offset of reference in code buffer
    bool relative;  // absolute or relative fixup
  };

  Extern(const string &symbol, Address address)
      : symbol(symbol), address(address) {}

  string symbol;           // symbolic name of external reference
  Address address;         // address of external reference
  std::vector<Ref> refs;   // references to symbol in code buffer
};

// A code generator emits machine code instructons into a buffer. If the
// provided buffer is null, the code generator allocates and grows its own
// buffer, and buffer_size determines the initial buffer size. The buffer is
// owned by the code generator and deallocated upon destruction of the code
// generator. If the provided buffer is not null, the code generator uses the
// provided buffer for code generation and assumes its size to be buffer_size.
// If the buffer is too small, a fatal error occurs. No deallocation of the
// buffer is done upon destruction of the code generator.
class CodeGenerator {
 public:
  CodeGenerator(void *buffer, int buffer_size);
  ~CodeGenerator();

  // Memory area for generated code.
  byte *begin() { return buffer_; }
  byte *end() { return pc_; }
  int size() { return pc_offset(); }

  // Current pc.
  Address pc() const { return pc_; }

  // Offset of pc in code buffer.
  int pc_offset() const { return static_cast<int>(pc_ - buffer_); }

  // Get the number of bytes available in the buffer.
  int available_space() const {
    return buffer_ + buffer_size_ - pc_;
  }

  // Increase size of code buffer.
  void GrowBuffer();

  // Bind label to current pc.
  void bind(Label *l);

  // Bind label to position.
  void bind_to(Label *l, int pos);

  // Check if there is not enough space available in the code buffer for
  // emitting one more instruction.
  bool buffer_overflow() const {
    return pc_ + kMaximumInstructionSize > buffer_ + buffer_size_;
  }

  // Get and set bytes in the code buffer.
  byte byte_at(int pos) { return buffer_[pos]; }
  void set_byte_at(int pos, byte value) { buffer_[pos] = value; }
  byte *addr_at(int pos)  { return buffer_ + pos; }
  uint32_t long_at(int pos) {
    return *reinterpret_cast<uint32_t *>(addr_at(pos));
  }
  void long_at_put(int pos, uint32_t x)  {
    *reinterpret_cast<uint32_t *>(addr_at(pos)) = x;
  }

  // Add external reference.
  void AddExtern(const string &symbol, Address address, bool relative = false);

  // List of external symbols in code buffer.
  const std::vector<Extern> &externs() const { return externs_; }

  static const int kMinimalBufferSize = 4096;
  static const int kMaximumInstructionSize = 32;

 protected:
  // The buffer into which code is generated. It could either be owned by the
  // code generator or be provided externally.
  byte *buffer_;
  int buffer_size_;
  bool own_buffer_;

  // The program counter, which points into the buffer above and moves forward.
  byte *pc_;

  // Internal reference positions, required for (potential) patching in
  // GrowBuffer(); contains only those internal references whose labels
  // are already bound.
  std::deque<int> refs_;

  // External symbols.
  std::vector<Extern> externs_;
};

// Helper class that ensures that there is enough space for generating
// instructions.  The constructor makes sure that there is enough space and (in
// debug mode) the destructor checks that we did not generate too much.
class EnsureSpace {
 public:
  explicit EnsureSpace(CodeGenerator *generator) : generator_(generator) {
    if (generator_->buffer_overflow()) generator_->GrowBuffer();
#ifdef DEBUG
    space_before_ = generator_->available_space();
#endif
  }

#ifdef DEBUG
  ~EnsureSpace() {
    int bytes_generated = space_before_ - generator_->available_space();
    DCHECK(bytes_generated < CodeGenerator::kMaximumInstructionSize);
  }
#endif

 private:
  CodeGenerator *generator_;
#ifdef DEBUG
  int space_before_;
#endif
};

// A code object holds a memory block of code that is executable.
class Code {
 public:
  // Initialize empty code object.
  Code() : memory_(nullptr), size_(0) {}

  // Initialize code object from memory block. This will make a copy of the
  // code object.
  Code(void *code, int size);

  // Initialize code object from generated code.
  Code(CodeGenerator *generator)
      : Code(generator->begin(), generator->size()) {}

  // Deallocate code block.
  ~Code();

  // Allocate executable memory for code object.
  void Allocate(void *code, int size);
  void Allocate(CodeGenerator *generator) {
    Allocate(generator->begin(), generator->size());
  }

  // Memory range for code block.
  byte *begin() const { return memory_; }
  byte *end() const { return memory_ + size_; }
  int size() const { return size_; }

  // Entry point for code block is assumed to be the beginning of the block.
  void *entry() const { return memory_; }

  // Execute code in code block.
  void Execute(void *arg) const {
    reinterpret_cast<void (*)(void *)>(entry())(arg);
  }
  uint64_t Execute(uint64_t arg1, uint64_t arg2) const {
    return reinterpret_cast<uint64_t (*)(uint64_t, uint64_t)>(
        entry())(arg1, arg2);
  }
  uint64_t Execute(const char *arg1, uint64_t arg2) const {
    return reinterpret_cast<uint64_t (*)(const char *arg1, uint64_t)>(
        entry())(arg1, arg2);
  }

 private:
  // Memory block for code block.
  byte *memory_;

  // Size of code block.
  int size_;
};

}  // namespace jit
}  // namespace sling

#endif  // JIT_CODEGEN_H_

