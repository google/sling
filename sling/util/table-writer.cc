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

#include "sling/util/table-writer.h"

#include "sling/file/file.h"
#include "sling/string/numbers.h"
#include "sling/string/printf.h"
#include "sling/string/strcat.h"

namespace sling {

int TableWriter::Table::GetRowOrDie(const string &row) {
  const auto it = rows.find(row);
  CHECK(it != rows.end()) << "Unknown row: " << row;
  return it->second;
}

int TableWriter::Table::GetColOrDie(const string &col) {
  const auto it = columns.find(col);
  CHECK(it != columns.end()) << "Unknown column: " << col;
  return it->second;
}

TableWriter::Cell *TableWriter::Table::GetOrMakeCell(int row, int col) {
  CHECK_GE(row, 0);
  CHECK_GE(col, 0);
  CHECK_LT(col, columns.size());
  if (cells.size() <= row) {
    int old_size = cells.size();
    cells.resize(row + 1);
    for (int i = old_size; i < row + 1; ++i) {
      cells[i].resize(columns.size());
    }
  }
  CHECK_LT(col, cells[row].size());
  return &cells[row][col];
}

void TableWriter::StartTable(const string &name) {
  tables_.emplace_back(name);
  current_ = &tables_.back();
}

void TableWriter::SetCommasInNumbers(bool commas) {
  CHECK(current_ != nullptr);
  current_->commas = commas;
}

void TableWriter::SetDecimalPlaces(int decimals) {
  CHECK(current_ != nullptr);
  current_->decimals = decimals;
}

void TableWriter::SetColumns(const std::vector<string> &names) {
  CHECK(current_ != nullptr);
  for (const string &name : names) {
    auto &c = current_->columns;
    const auto it = c.find(name);
    CHECK(it == c.end()) << "Duplicate column: " << name;
    int index = c.size();
    c[name] = index;
  }
}

void TableWriter::AddNamedRow(const string &name) {
  CHECK(current_ != nullptr);
  const auto it = current_->rows.find(name);
  CHECK(it == current_->rows.end()) << "Row " << name << " already exists";
  int index = current_->cells.size();
  current_->rows[name] = index;
  current_->cells.emplace_back();
  current_->cells.back().resize(current_->columns.size());
}

void TableWriter::AddToCell(int row, int col, int64 value) {
  CHECK(current_ != nullptr);
  current_->GetOrMakeCell(row, col)->Add(value);
}

void TableWriter::AddToCell(int row, int col, int value) {
  AddToCell(row, col, static_cast<int64>(value));
}

void TableWriter::AddToCell(const string &row, const string &col, int64 value) {
  CHECK(current_ != nullptr);
  int r = current_->GetRowOrDie(row);
  int c = current_->GetColOrDie(col);
  AddToCell(r, c, value);
}

void TableWriter::AddToCell(const string &row, const string &col, int value) {
  AddToCell(row, col, static_cast<int64>(value));
}

void TableWriter::AddToCell(const string &row, int col, int64 value) {
  CHECK(current_ != nullptr);
  int r = current_->GetRowOrDie(row);
  AddToCell(r, col, value);
}

void TableWriter::AddToCell(const string &row, int col, int value) {
  AddToCell(row, col, static_cast<int64>(value));
}

void TableWriter::AddToCell(int row, const string &col, int64 value) {
  CHECK(current_ != nullptr);
  int c = current_->GetColOrDie(col);
  AddToCell(row, c, value);
}

void TableWriter::AddToCell(int row, const string &col, int value) {
  AddToCell(row, col, static_cast<int64>(value));
}

void TableWriter::AddToCell(int row, int col, float value) {
  CHECK(current_ != nullptr);
  current_->GetOrMakeCell(row, col)->Add(value);
}

void TableWriter::AddToCell(const string &row, const string &col, float value) {
  CHECK(current_ != nullptr);
  int r = current_->GetRowOrDie(row);
  int c = current_->GetColOrDie(col);
  AddToCell(r, c, value);
}

void TableWriter::AddToCell(const string &row, int col, float value) {
  CHECK(current_ != nullptr);
  int r = current_->GetRowOrDie(row);
  AddToCell(r, col, value);
}

void TableWriter::AddToCell(int row, const string &col, float value) {
  CHECK(current_ != nullptr);
  int c = current_->GetColOrDie(col);
  AddToCell(row, c, value);
}

void TableWriter::SetCell(int row, int col, int64 value) {
  CHECK(current_ != nullptr);
  current_->GetOrMakeCell(row, col)->Set(value);
}

void TableWriter::SetCell(int row, int col, int value) {
  SetCell(row, col, static_cast<int64>(value));
}

void TableWriter::SetCell(const string &row, const string &col, int64 value) {
  CHECK(current_ != nullptr);
  int r = current_->GetRowOrDie(row);
  int c = current_->GetColOrDie(col);
  SetCell(r, c, value);
}

void TableWriter::SetCell(const string &row, const string &col, int value) {
  SetCell(row, col, static_cast<int64>(value));
}

void TableWriter::SetCell(const string &row, int col, int64 value) {
  CHECK(current_ != nullptr);
  int r = current_->GetRowOrDie(row);
  SetCell(r, col, value);
}

void TableWriter::SetCell(const string &row, int col, int value) {
  SetCell(row, col, static_cast<int64>(value));
}

void TableWriter::SetCell(int row, const string &col, int64 value) {
  CHECK(current_ != nullptr);
  int c = current_->GetColOrDie(col);
  SetCell(row, c, value);
}

void TableWriter::SetCell(int row, const string &col, int value) {
  SetCell(row, col, static_cast<int64>(value));
}

void TableWriter::SetCell(int row, int col, float value) {
  CHECK(current_ != nullptr);
  current_->GetOrMakeCell(row, col)->Set(value);
}

void TableWriter::SetCell(const string &row, const string &col, float value) {
  CHECK(current_ != nullptr);
  int r = current_->GetRowOrDie(row);
  int c = current_->GetColOrDie(col);
  SetCell(r, c, value);
}

void TableWriter::SetCell(const string &row, int col, float value) {
  CHECK(current_ != nullptr);
  int r = current_->GetRowOrDie(row);
  SetCell(r, col, value);
}

void TableWriter::SetCell(int row, const string &col, float value) {
  CHECK(current_ != nullptr);
  int c = current_->GetColOrDie(col);
  SetCell(row, c, value);
}

void TableWriter::SetCell(int row, int col, const string &value) {
  CHECK(current_ != nullptr);
  current_->GetOrMakeCell(row, col)->Set(value);
}

void TableWriter::SetCell(
    const string &row, const string &col, const string &value) {
  CHECK(current_ != nullptr);
  int r = current_->GetRowOrDie(row);
  int c = current_->GetColOrDie(col);
  SetCell(r, c, value);
}

void TableWriter::SetCell(const string &row, int col, const string &value) {
  CHECK(current_ != nullptr);
  int r = current_->GetRowOrDie(row);
  SetCell(r, col, value);
}

void TableWriter::SetCell(int row, const string &col, const string &value) {
  CHECK(current_ != nullptr);
  int c = current_->GetColOrDie(col);
  SetCell(row, c, value);
}

void TableWriter::Annotate(int row, int col, const string &value) {
  CHECK(current_ != nullptr);
  current_->GetOrMakeCell(row, col)->annotation = value;
}

void TableWriter::Annotate(
    const string &row, const string &col, const string &value) {
  CHECK(current_ != nullptr);
  int r = current_->GetRowOrDie(row);
  int c = current_->GetColOrDie(col);
  Annotate(r, c, value);
}

void TableWriter::Annotate(const string &row, int col, const string &value) {
  CHECK(current_ != nullptr);
  int r = current_->GetRowOrDie(row);
  Annotate(r, col, value);
}

void TableWriter::Annotate(int row, const string &col, const string &value) {
  CHECK(current_ != nullptr);
  int c = current_->GetColOrDie(col);
  Annotate(row, c, value);
}

void TableWriter::AddRow(const string &cell1, const string &cell2) {
  CHECK(current_ != nullptr);
  int row = current_->cells.size();
  current_->GetOrMakeCell(row, 0)->Set(cell1);
  current_->GetOrMakeCell(row, 1)->Set(cell2);
}

void TableWriter::AddRow(const string &cell1, int64 cell2) {
  CHECK(current_ != nullptr);
  int row = current_->cells.size();
  current_->GetOrMakeCell(row, 0)->Set(cell1);
  current_->GetOrMakeCell(row, 1)->Set(cell2);
}

void TableWriter::AddRow(const string &cell1, int cell2) {
  AddRow(cell1, static_cast<int64>(cell2));
}

void TableWriter::AddRow(const string &cell1, float cell2) {
  CHECK(current_ != nullptr);
  int row = current_->cells.size();
  current_->GetOrMakeCell(row, 0)->Set(cell1);
  current_->GetOrMakeCell(row, 1)->Set(cell2);
}

void TableWriter::AddRow(int64 cell1, const string &cell2) {
  CHECK(current_ != nullptr);
  int row = current_->cells.size();
  current_->GetOrMakeCell(row, 0)->Set(cell1);
  current_->GetOrMakeCell(row, 1)->Set(cell2);
}

void TableWriter::AddRow(int cell1, const string &cell2) {
  AddRow(static_cast<int64>(cell1), cell2);
}

void TableWriter::AddRow(int64 cell1, int64 cell2) {
  CHECK(current_ != nullptr);
  int row = current_->cells.size();
  current_->GetOrMakeCell(row, 0)->Set(cell1);
  current_->GetOrMakeCell(row, 1)->Set(cell2);
}

void TableWriter::AddRow(int64 cell1, float cell2) {
  CHECK(current_ != nullptr);
  int row = current_->cells.size();
  current_->GetOrMakeCell(row, 0)->Set(cell1);
  current_->GetOrMakeCell(row, 1)->Set(cell2);
}

void TableWriter::AddRow(int cell1, float cell2) {
  AddRow(static_cast<int64>(cell1), cell2);
}

void TableWriter::AddRow(float cell1, const string &cell2) {
  CHECK(current_ != nullptr);
  int row = current_->cells.size();
  current_->GetOrMakeCell(row, 0)->Set(cell1);
  current_->GetOrMakeCell(row, 1)->Set(cell2);
}

void TableWriter::AddRow(float cell1, int64 cell2) {
  CHECK(current_ != nullptr);
  int row = current_->cells.size();
  current_->GetOrMakeCell(row, 0)->Set(cell1);
  current_->GetOrMakeCell(row, 1)->Set(cell2);
}

void TableWriter::AddRow(float cell1, int cell2) {
  AddRow(cell1, static_cast<int64>(cell2));
}

void TableWriter::AddRow(float cell1, float cell2) {
  CHECK(current_ != nullptr);
  int row = current_->cells.size();
  current_->GetOrMakeCell(row, 0)->Set(cell1);
  current_->GetOrMakeCell(row, 1)->Set(cell2);
}

void TableWriter::SwitchToTable(int index) {
  CHECK_GE(index, 0);
  CHECK_LT(index, tables_.size());
  current_= &tables_[index];
}

void TableWriter::SwitchToTable(const string &name) {
  current_= nullptr;
  for (Table &table : tables_) {
    if (table.name == name) {
      current_ = &table;
      break;
    }
  }
  CHECK(current_ != nullptr) << "No such table: " << name;
}

void TableWriter::Write(string *contents) const {
  static const string kColSeparator = " || ";
  static const string kNotFilled = "-";
  for (const Table &table : tables_) {
    StrAppend(contents, table.name, "\n");

    std::vector<int> lengths;
    std::vector<string> names;
    int num_cols = table.columns.size();
    lengths.resize(num_cols, 0);
    names.resize(num_cols);
    for (const auto &kv : table.columns) {
      int index = kv.second;
      names[index] = kv.first;
      lengths[index] = kv.first.size();
    }

    std::vector<string> cells;
    for (const auto &row : table.cells) {
      for (int i = 0; i < num_cols; ++i) {
        const auto &cell = row[i];
        string s;
        switch (cell.type) {
          case Cell::NONE:
            s = kNotFilled;
            break;
          case Cell::INT:
            s = table.commas ?
                SimpleItoaWithCommas(cell.int_value) : StrCat(cell.int_value);
            break;
          case Cell::FLOAT: {
            string format = StrCat("%.", table.decimals, "f");
            s = StringPrintf(format.c_str(), cell.float_value);
            break;
          }
          case Cell::STRING:
           s = cell.str_value;
           break;
         default:
           break;
        }
        if (!cell.annotation.empty()) s = StrCat(cell.annotation, s);
        cells.push_back(s);
        if (lengths[i] < s.size()) lengths[i] = s.size();
      }
    }

    // Write column headers.
    int total = 0;
    for (int len : lengths) {
      total += len + kColSeparator.size();
    }
    total -= kColSeparator.size();
    string line(total, '=');
    StrAppend(contents, line, "\n");
    for (int col = 0; col < names.size(); ++col) {
      int left_padding = lengths[col] - names[col].size();
      string padding = left_padding > 0 ? string(left_padding, ' ') : "";
      StrAppend(contents, padding, names[col]);
      if (col < names.size() - 1) StrAppend(contents, kColSeparator);
    }
    StrAppend(contents, "\n", line, "\n");

    // Write the cells.
    int i = 0;
    for (const string &cell : cells) {
      int col = i % num_cols;
      int left_padding = lengths[col] - cell.size();
      string padding = left_padding > 0 ? string(left_padding, ' ') : "";
      StrAppend(contents, padding, cell);
      StrAppend(contents, (col < num_cols - 1) ? kColSeparator : "\n");
      i++;
    }
    StrAppend(contents, line, "\n\n");
  }
}

void TableWriter::Write(const string &file) const {
  string contents;
  Write(&contents);
  CHECK(File::WriteContents(file, contents));
}

}  // namespace sling

