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

#include "sling/util/elf-writer.h"

#include <elf.h>
#include <string.h>
#include <stdio.h>

namespace sling {

// Offset for global symbol indices.
static const int GLOBAL_SYMBOL_OFFSET = 1 << 20;

Elf::Elf() {
  // Initialize header.
  memset(&ehdr_, 0, sizeof(ehdr_));
  ehdr_.e_ident[EI_MAG0] = ELFMAG0;
  ehdr_.e_ident[EI_MAG1] = ELFMAG1;
  ehdr_.e_ident[EI_MAG2] = ELFMAG2;
  ehdr_.e_ident[EI_MAG3] = ELFMAG3;
  ehdr_.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr_.e_ident[EI_DATA] = ELFDATA2LSB;
  ehdr_.e_ident[EI_VERSION] = EV_CURRENT;
  ehdr_.e_ident[EI_OSABI] = ELFOSABI_SYSV;
  ehdr_.e_type = ET_REL;
  ehdr_.e_machine = EM_X86_64;
  ehdr_.e_version = EV_CURRENT;
  ehdr_.e_ehsize = sizeof(ehdr_);
  ehdr_.e_shoff = sizeof(ehdr_);
  ehdr_.e_shentsize = sizeof(Elf64_Shdr);

  // Add null symbol as first symbol.
  AddSymbol("", false);

  // Add null section as first section.
  AddSection("", SHT_NULL);

  // Add section for symbol table.
  symtab_ = AddSection(".symtab", SHT_SYMTAB);
}

Elf::~Elf() {
  for (auto *section : sections_) delete section;
  for (auto *symbol : local_symbols_) delete symbol;
  for (auto *symbol : global_symbols_) delete symbol;
}

Elf::Section *Elf::AddSection(const char *name, Elf64_Word type) {
  // Allocate new section.
  Section *section = new Section(sections_.size());
  sections_.push_back(section);

  // Set section name.
  int namelen = strlen(name);
  section->hdr.sh_name = section_names_.size();
  section_names_.append(name, namelen + 1);

  // Set section type.
  section->hdr.sh_type = type;
  if (type != SHT_NULL) section->hdr.sh_addralign = 1;

  // Add symbol for section.
  if (type == SHT_PROGBITS) {
    section->symidx = AddSymbol("", section, STB_LOCAL, STT_SECTION)->index;
  }

  return section;
}

Elf::Symbol *Elf::AddSymbol(const char *name, bool global) {
  // Allocate new symbol.
  Symbol *symbol;
  if (global) {
    symbol = new Symbol(global_symbols_.size() + GLOBAL_SYMBOL_OFFSET);
    global_symbols_.push_back(symbol);
  } else {
    symbol = new Symbol(local_symbols_.size());
    local_symbols_.push_back(symbol);
  }

  // Set symbol name.
  int namelen = strlen(name);
  if (namelen == 0 && !symbol_names_.empty()) {
    symbol->sym.st_name = 0;
  } else {
    symbol->sym.st_name = symbol_names_.size();
    symbol_names_.append(name, namelen + 1);
  }

  return symbol;
}

Elf::Symbol *Elf::AddSymbol(const char *name, Section *section,
                            int bind, int type, int size, int value) {
  Symbol *symbol = AddSymbol(name, bind == STB_GLOBAL);
  symbol->sym.st_info = ELF64_ST_INFO(bind, type);
  if (section != nullptr) {
    symbol->sym.st_shndx = section->index;
  }
  symbol->sym.st_size = size;
  symbol->sym.st_value = value;
  return symbol;
}

void Elf::Update() {
  // Build symbol table. Local symbols must come before non-local symbols in the
  // symbol table.
  for (int i = 0; i < local_symbols_.size(); ++i) {
    symbol_data_.push_back(local_symbols_[i]->sym);
  }
  for (int i = 0; i < global_symbols_.size(); ++i) {
    symbol_data_.push_back(global_symbols_[i]->sym);
  }

  symtab_->data = symbol_data_.data();
  symtab_->hdr.sh_size = symbol_data_.size() * sizeof(Elf64_Sym);
  symtab_->hdr.sh_entsize = sizeof(Elf64_Sym);
  symtab_->hdr.sh_addralign = 8;
  symtab_->hdr.sh_info = local_symbols_.size();

  // Build symbol name string table.
  Section *strtab = AddSection(".strtab", SHT_STRTAB);
  strtab->hdr.sh_size = symbol_names_.size();
  strtab->data = symbol_names_.data();
  symtab_->hdr.sh_link = strtab->index;

  // Build section name string table.
  Section *shstrtab = AddSection(".shstrtab", SHT_STRTAB);
  shstrtab->hdr.sh_size = section_names_.size();
  shstrtab->data = section_names_.data();

  // Set number of sections in header.
  ehdr_.e_shstrndx = shstrtab->index;
  ehdr_.e_shnum = sections_.size();
}

void Elf::Write(const char *filename) {
  // Open output file.
  FILE *f = fopen(filename, "w");
  if (!f) {
    perror(filename);
    abort();
  }

  // Write ELF header.
  fwrite(&ehdr_, 1, sizeof(Elf64_Ehdr), f);

  // Write section headers.
  int offset = sizeof(Elf64_Ehdr) + ehdr_.e_shnum * sizeof(Elf64_Shdr);
  for (Section *section : sections_) {
    if (section->data != nullptr) {
      section->hdr.sh_offset = offset;
      offset += section->hdr.sh_size;
    }
    fwrite(&section->hdr, 1, sizeof(Elf64_Shdr), f);
  }

  // Write section data.
  for (Section *section : sections_) {
    if (section->data != nullptr) {
      fwrite(section->data, 1, section->hdr.sh_size, f);
    }
  }

  // Close output file.
  fclose(f);
}

Elf::Buffer::Buffer(Elf *elf,
                    const char *name,
                    const char *relaname,
                    Elf64_Word type,
                    Elf64_Word flags) : elf(elf) {
  progbits = elf->AddSection(name, type);
  progbits->hdr.sh_flags = flags;
  if (relaname) {
    rela = elf->AddSection(relaname, SHT_RELA);
    rela->hdr.sh_link = elf->symtab()->index;
    rela->hdr.sh_info = progbits->index;
    rela->hdr.sh_entsize = sizeof(Elf64_Rela);
    rela->hdr.sh_addralign = 8;
  } else {
    rela = nullptr;
  }
}

void Elf::Buffer::Add(const void *data, int size) {
  if (data != nullptr) {
    content.append(reinterpret_cast<const char *>(data), size);
  } else {
    content.resize(content.size() + size);
  }
}

void Elf::Buffer::Add8(uint8_t data) {
  Add(&data, sizeof(uint8_t));
}

void Elf::Buffer::Add32(uint32_t data) {
  Add(&data, sizeof(uint32_t));
}

void Elf::Buffer::Add64(uint64_t data) {
  Add(&data, sizeof(uint64_t));
}

void Elf::Buffer::AddPtr(Buffer *buffer, int offset) {
  AddReloc(buffer, R_X86_64_64, offset);
  Add64(0);
}

void Elf::Buffer::AddPtr32(Buffer *buffer, int offset) {
  AddReloc(buffer, R_X86_64_32, offset);
  Add32(0);
}

void Elf::Buffer::AddExternPtr(Elf::Symbol *symbol) {
  AddReloc(symbol, R_X86_64_64);
  Add64(0);
}

int32_t Elf::Buffer::Clear32(int offset) {
  int32_t *ptr = reinterpret_cast<int32_t *>(&content[offset]);
  int32_t prev = *ptr;
  *ptr = 0;
  return prev;
}

int64_t Elf::Buffer::Clear64(int offset) {
  int64_t *ptr = reinterpret_cast<int64_t *>(&content[offset]);
  int64_t prev = *ptr;
  *ptr = 0;
  return prev;
}

void Elf::Buffer::Align(int alignment) {
  // Pad section buffer.
  while (offset() % alignment != 0) Add8(0);

  // Update section alignment.
  if (progbits->hdr.sh_addralign < alignment) {
    progbits->hdr.sh_addralign = alignment;
  }
}

void Elf::Buffer::AddReloc(Section *section, int type, int addend) {
  Elf64_Rela rel;
  rel.r_offset = offset();
  rel.r_info = ELF64_R_INFO(section->symidx, type);
  rel.r_addend = addend;
  relocs.push_back(rel);
}

void Elf::Buffer::AddReloc(Symbol *symbol, int type, int addend, int offset) {
  Elf64_Rela rel;
  rel.r_offset = offset;
  rel.r_info = ELF64_R_INFO(symbol->index, type);
  rel.r_addend = addend;
  relocs.push_back(rel);
}

void Elf::Buffer::AddReloc(Symbol *symbol, int type, int addend) {
  AddReloc(symbol, type, addend, offset());
}

void Elf::Buffer::AddReloc(Buffer *buffer, int type, int addend) {
  AddReloc(buffer->progbits, type, addend);
}

void Elf::Buffer::Update() {
  if (progbits->hdr.sh_type != SHT_NOBITS) {
    progbits->data = content.data();
  }
  progbits->hdr.sh_size = content.size();
  if (rela) {
    // Adjust symbol indices for global symbols in relocations.
    int global_offset = elf->num_local_symbols() - GLOBAL_SYMBOL_OFFSET;
    for (Elf64_Rela &rel : relocs) {
      int symidx = ELF64_R_SYM(rel.r_info);
      if (symidx >=  GLOBAL_SYMBOL_OFFSET) {
        rel.r_info = ELF64_R_INFO(symidx + global_offset,
                                  ELF64_R_TYPE(rel.r_info));
      }
    }

    // Set data for relocation section.
    rela->data = relocs.data();
    rela->hdr.sh_size = relocs.size() * sizeof(Elf64_Rela);
  }
}

}  // namespace sling

