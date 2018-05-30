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

#ifndef SLING_MYELIN_ELF_LINKER_H_
#define SLING_MYELIN_ELF_LINKER_H_

#include <string>
#include <unordered_map>

#include "sling/base/types.h"
#include "sling/myelin/compute.h"
#include "sling/util/elf-writer.h"

namespace sling {
namespace myelin {

// Myelin linker for outputting code and data to ELF object file.
class ElfLinker : public Linker {
 public:
  struct Options {
    // Generate data section with model parameters
    bool generate_data = false;

    // Generate writeable uninitalized data section with model parameters.
    bool writeable_data = false;

    // Code alignment.
    int code_align = 16;
  };

  ElfLinker(const Options &options) : options_(options) {}
  ElfLinker() {}

  // Start generating code for cell.
  void BeginCell(Cell *cell) override;

  // Add local entry point for step.
  void EndStep(Step *step, int offset) override;

  // Add code for cell.
  void EndCell(Cell *cell,
               jit::CodeGenerator *generator,
               jit::Code *code,
               int data_size) override;

  // Add data for tensor.
  void AddData(Tensor *data) override;

  // Add device code for step.
  void AddDeviceCode(Step *step, const string &code) override;

  // Link sections.
  void Link();

  // Write object file.
  void Write(const char *filename);

 private:
  // Linker options.
  Options options_;

  // ELF object file writer.
  Elf elf_;

  // Code section.
  Elf::Buffer code_{&elf_, ".text", ".rela.text",
                    SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR};

  // Read-only data section.
  Elf::Buffer rodata_{&elf_, ".rodata", nullptr, SHT_PROGBITS, SHF_ALLOC};

  // Uninitialized data section.
  Elf::Buffer bss_{&elf_, ".bss", nullptr, SHT_NOBITS, SHF_ALLOC | SHF_WRITE};

  // External symbols.
  std::unordered_map<string, Elf::Symbol *> symbols_;
};

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_ELF_LINKER_H_

