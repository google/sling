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

#ifndef SLING_MYELIN_AOT_LINKER_H_
#define SLING_MYELIN_AOT_LINKER_H_

#include <ctype.h>
#include <iostream>
#include <sstream>
#include <string>

#include "sling/base/types.h"
#include "sling/myelin/compute.h"
#include "sling/util/elf-writer.h"

namespace sling {
namespace myelin {

// Myelin linker for ahead-of-time compilation. This produces an ELF object
// file with the generated code (and data), as well as a C++ header file for
// accessing the model. The model data can either be stored in the object file
// itself in the read-only section for initialized data (.rodata), or be stored
// externally so it can be loaded at run time, in which case space is allocated
// in the unintialized data section (.bss).
class AOTLinker : public Linker {
 public:
  struct Options {
    string flow_file;              // source flow file
    bool external_data = false;    // parameter data stored in external file
    bool uppercase_names = false;  // uppercase class names
    string ns;                     // C++ name space for generated code
  };

  // Initialize ahead-of-time linker from linker options.
  AOTLinker(const Options &options);

  // Linker interface.
  void BeginCell(Cell *cell) override;
  void EndCell(Cell *cell,
               jit::CodeGenerator *generator,
               jit::Code *code,
               int data_size) override;
  void AddData(Tensor *data) override;

  // Add channel.
  void AddChannel(const string &name, Tensor *format);

  // Link sections.
  void Link();

  // Write ELF object file.
  void Write(const string &filename);

  // Write header file.
  void WriteHeader(const string &filename);

  // Write data file.
  void WriteData(const string &filename);

 private:
  // Return sanitized name that is a legal C++ identifier.
  string Sanitized(const string &name);

  // Return sanitized class name that is a legal C++ identifier.
  string SanitizedClassName(const string &name);

  // Return mangled symbol for name.
  string Mangled(const string &name, bool func);

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

  // Output stream for C++ header file.
  std::ostringstream header_;
};

}  // namespace myelin
}  // namespace sling

#endif  // SLING_MYELIN_AOT_LINKER_H_

