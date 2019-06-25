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

#include <iostream>

#include "sling/myelin/elf-linker.h"

namespace sling {
namespace myelin {

void ElfLinker::BeginCell(Cell *cell) {
  // Align code buffer before generating new cell computation function.
  code_.Align(options_.code_align);
}

void ElfLinker::EndStep(Step *step, int offset) {
  // Add entry point for step.
  if (!step->noop()) {
    string name = step->name() + ":" + step->kernel()->Name();
    elf_.AddSymbol(name.c_str(), code_.progbits, STB_LOCAL, STT_FUNC,
                   0, code_.offset() + offset);
  }
}

void ElfLinker::EndCell(Cell *cell,
                        jit::CodeGenerator *generator,
                        jit::Code *code,
                        int data_size) {
  // Add entry point for cell computation.
  int code_size = generator->size() - data_size;
  elf_.AddSymbol(cell->name().c_str(), code_.progbits, STB_GLOBAL, STT_FUNC,
                 code_size, code_.offset());

  // Add symbol for constant data.
  if (data_size > 0) {
    string data_name = cell->name() + "_data";
    elf_.AddSymbol(data_name.c_str(), code_.progbits, STB_LOCAL, STT_OBJECT,
                   data_size, code_.offset() + code_size);
  }

  // Output code to code section.
  int code_start = code_.offset();
  code_.Add(generator->begin(), generator->size());

  // Add relocations for external references.
  for (auto &e : generator->externs()) {
    // Try to find existing symbol in object file. If symbol is not known,
    // a new undefined symbol is added.
    Elf::Symbol *sym = symbols_[e.symbol];
    if (sym == nullptr) {
      sym = elf_.AddSymbol(e.symbol.c_str(), nullptr,
                           STB_GLOBAL, STT_NOTYPE);
      symbols_[e.symbol] = sym;
    }

    // Add relocations to code.
    for (auto &ref : e.refs) {
      if (ref.relative) {
        int addend = code_.Clear32(code_start + ref.offset);
        code_.AddReloc(sym, R_X86_64_PC32, addend, code_start + ref.offset);
      } else {
        code_.AddReloc(sym, R_X86_64_64, 0, code_start + ref.offset);
        code_.Clear64(code_start + ref.offset);
      }
    }
  }

  // Generate JIT code object as well.
  code->Allocate(generator);
}

void ElfLinker::AddData(Tensor *data) {
  if (options_.generate_data) {
    // Determine section for data.
    auto *s = options_.writeable_data ? &bss_ : &rodata_;

    // Ensure alignment of tensor data.
    s->Align(data->byte_alignment());

    // Add symbol for data block.
    Elf::Symbol *sym = elf_.AddSymbol(data->name().c_str(), s->progbits,
                                      STB_LOCAL, STT_OBJECT,
                                      data->space(), s->offset());
    symbols_[data->name()] = sym;

    // Output tensor to data section.
    s->Add(data->data(), data->space());
  }
}

void ElfLinker::AddDeviceCode(Step *step, const string &code) {
  std::cout << "GPU code for step " << step->name() << ":\n" << code << "\n";
}

void ElfLinker::Link() {
  code_.Update();
  rodata_.Update();
  bss_.Update();
  elf_.Update();
}

void ElfLinker::Write(const char *filename) {
  elf_.Write(filename);
}

}  // namespace myelin
}  // namespace sling

