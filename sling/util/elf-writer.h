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

#ifndef SLING_UTIL_ELF_WRITER_
#define SLING_UTIL_ELF_WRITER_

#include <elf.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

namespace sling {

// ELF object file writer.
class Elf {
 public:
  // Section in ELF file.
  struct Section {
    Section(int idx) {
      memset(&hdr, 0, sizeof(Elf64_Shdr));
      index = idx;
    }
    Elf64_Shdr hdr;
    int index;
    int symidx = 0;
    const void *data = nullptr;
  };

  // Symbol in ELF file.
  struct Symbol {
    Symbol(int idx) {
      memset(&sym, 0, sizeof(Elf64_Sym));
      index = idx;
    }
    Elf64_Sym sym;
    int index;
  };

  // Buffer for generating section.
  struct Buffer {
    Buffer(Elf *elf, const char *name, const char *relaname,
           Elf64_Word type, Elf64_Word flags);

    // Add data to section buffer.
    void Add(const void *data, int size);
    void Add8(uint8_t data);
    void Add32(uint32_t data);
    void Add64(uint64_t data);
    void AddPtr(Buffer *buffer, int offset);
    void AddPtr32(Buffer *buffer, int offset);
    void AddExternPtr(Elf::Symbol *symbol);

    // Clear dword/qword at offset in section buffer and return previous value.
    int32_t Clear32(int offset);
    int64_t Clear64(int offset);

    // Pad buffer to alignment.
    void Align(int alignment);

    // Add relocation to section.
    void AddReloc(Section *section, int type, int addend = 0);
    void AddReloc(Symbol *symbol, int type, int addend, int offset);
    void AddReloc(Symbol *symbol, int type, int addend = 0);
    void AddReloc(Buffer *buffer, int type, int addend = 0);

    // Update section information. This must be called once after all the data
    // has been added to the section buffer and before the ELF file is updated.
    void Update();

    // Return current offset in section buffer.
    int offset() const { return content.size(); }

    Elf *elf;                        // ELF file for section
    Section *progbits;               // section for data
    Section *rela;                   // section for relocations
    std::string content;             // section content
    std::vector<Elf64_Rela> relocs;  // section relocations
  };

  Elf();
  ~Elf();

  // Add section to ELF file.
  Section *AddSection(const char *name, Elf64_Word type);

  // Add symbol to ELF file.
  Symbol *AddSymbol(const char *name, bool global);
  Symbol *AddSymbol(const char *name, Section *section,
                    int bind, int type, int size = 0, int value = 0);

  // Update symbol and section tables. The must be called once before the ELF
  // file is written out.
  void Update();

  // Write ELF object file.
  void Write(const char *filename);

  // Return symbol table.
  Section *symtab() { return symtab_; }

  // Return number of local symbols.
  int num_local_symbols() { return local_symbols_.size(); }

 private:
  // ELF file header.
  Elf64_Ehdr ehdr_;

  // Local and global symbols.
  std::vector<Symbol *> local_symbols_;
  std::vector<Symbol *> global_symbols_;

  // Symbol names.
  std::string symbol_names_;

  // Symbol table section.
  Section *symtab_;

  // Symbol section contents.
  std::vector<Elf64_Sym> symbol_data_;

  // Sections.
  std::vector<Section *> sections_;

  // Section names.
  std::string section_names_;
};

}  // namespace sling

#endif  // SLING_UTIL_ELF_WRITER_
