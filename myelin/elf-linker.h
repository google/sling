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

#ifndef MYELIN_ELF_LINKER_H_
#define MYELIN_ELF_LINKER_H_

#include <string>
#include <unordered_map>

#include "base/types.h"
#include "myelin/compute.h"
#include "util/elf-writer.h"

namespace sling {
namespace myelin {

// Myelin linker for outputting code and data to ELF object file.
class ElfLinker : public Linker {
 public:
  // Start generating code for cell.
  void BeginCell(Cell *cell) override;

  // Add local entry point for step.
  void AddStep(Step *step, int offset) override;

  // Add code for cell.
  void EndCell(Cell *cell,
               jit::CodeGenerator *generator,
               jit::Code *code,
               int data_size) override;

  // Add data for tensor.
  void AddData(Tensor *data) override;

  // Link sections.
  void Link();

  // Write object file.
  void Write(const char *filename);

  // Set data generation flag.
  void set_generate_data(bool b) { generate_data_ = b; }

 private:
  // ELF object file writer.
  Elf elf_;

  // Code section.
  Elf::Buffer code_{&elf_, ".text", ".rela.text",
                    SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR};

  // Data section.
  Elf::Buffer data_{&elf_, ".rodata", ".rela.rodata",
                    SHT_PROGBITS, SHF_ALLOC};

  // External symbols.
  std::unordered_map<string, Elf::Symbol *> symbols_;

  // Generate data sections for tensor data.
  bool generate_data_ = false;
};

}  // namespace myelin
}  // namespace sling

#endif  // MYELIN_ELF_LINKER_H_

