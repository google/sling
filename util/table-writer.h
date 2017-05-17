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

#ifndef UTIL_TABLE_WRITER_H_
#define UTIL_TABLE_WRITER_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "base/logging.h"
#include "base/macros.h"
#include "base/types.h"

namespace sling {

// Utility class for pretty printing multiple tables.
class TableWriter {
 public:
  // Starts a new table. Row and column manipulation methods from now on
  // will affect this new table.
  void StartTable(const string &name);

  // Sets the names of the table columns.
  void SetColumns(const std::vector<string> &names);

  // Whether commas should be used while printing integers in the current table.
  void SetCommasInNumbers(bool commas);

  // Number of decimal places to use while printing floats in the current table.
  void SetDecimalPlaces(int decimals);

  // Adds a row with the specified name. The name just a handy alias that
  // can be used in subsequent AddToCell()/SetCell() calls.
  // Check fails if the name already exists.
  void AddNamedRow(const string &row_name);

  // Adds 'value' to the existing cell in the row named 'row' and column
  // named 'col'. If no such cell exists, then it is created and initialized
  // to 'value'.
  // Check fails if the existing cell has a different type than 'value'.
  // 'int' and 'int64' are considered the same, but 'int' and 'float' aren't.
  void AddToCell(const string &row, const string &col, int value);
  void AddToCell(const string &row, const string &col, int64 value);
  void AddToCell(const string &row, const string &col, float value);

  // Other flavors that use zero-based indices instead of row/column names.
  // A column index needs to be in range, but a row index can be out of range.
  // In this case, empty intervening rows will be added to the table.
  void AddToCell(int row, const string &col, int value);
  void AddToCell(int row, const string &col, int64 value);
  void AddToCell(int row, const string &col, float value);
  void AddToCell(int row, int col, int value);
  void AddToCell(int row, int col, int64 value);
  void AddToCell(int row, int col, float value);
  void AddToCell(const string &row, int col, int value);
  void AddToCell(const string &row, int col, int64 value);
  void AddToCell(const string &row, int col, float value);

  // Sets the specified set to the given value, overriding any previous value.
  void SetCell(const string &row, const string &col, int value);
  void SetCell(const string &row, const string &col, int64 value);
  void SetCell(const string &row, const string &col, float value);
  void SetCell(int row, const string &col, int value);
  void SetCell(int row, const string &col, int64 value);
  void SetCell(int row, const string &col, float value);
  void SetCell(int row, int col, int value);
  void SetCell(int row, int col, int64 value);
  void SetCell(int row, int col, float value);
  void SetCell(const string &row, int col, int value);
  void SetCell(const string &row, int col, int64 value);
  void SetCell(const string &row, int col, float value);

  // Versions of SetCell() for string-valued cells.
  void SetCell(const string &row, const string &col, const string &value);
  void SetCell(int row, const string &col, const string &value);
  void SetCell(int row, int col, const string &value);
  void SetCell(const string &row, int col, const string &value);

  // Annotate the specified cell. The annotation will be printed as a prefix
  // of the cell value.
  void Annotate(const string &row, const string &col, const string &annotation);
  void Annotate(const string &row, int col, const string &annotation);
  void Annotate(int row, int col, const string &annotation);
  void Annotate(int row, const string &col, const string &annotation);

  // Short-cut methods for 2-column tables.
  void AddRow(const string &cell1, const string &cell2);
  void AddRow(const string &cell1, int cell2);
  void AddRow(const string &cell1, int64 cell2);
  void AddRow(const string &cell1, float cell2);
  void AddRow(int cell1, const string &cell2);
  void AddRow(int64 cell1, const string &cell2);
  void AddRow(int64 cell1, int64 cell2);
  void AddRow(int cell1, float cell2);
  void AddRow(int64 cell1, float cell2);
  void AddRow(float cell1, const string &cell2);
  void AddRow(float cell1, int cell2);
  void AddRow(float cell1, int64 cell2);
  void AddRow(float cell1, float cell2);

  // Go back to a previously created table with the given name. Check fails
  // if such a table doesn't exist.
  void SwitchToTable(const string &table_name);
  void SwitchToTable(int table_index);

  // Pretty-print all the tables to 'contents'.
  void Write(string *contents) const;

  // Pretty-print all the tables to 'file'.
  void Write(const string &file) const;

 private:
  // Cell in a table.
  struct Cell {
    enum Type {INT, FLOAT, STRING, NONE};

    // Type of the cell's value.
    Type type = NONE;

    // Type-specific value of the cell.
    int64 int_value = 0;
    float float_value = 0;
    string str_value;

    // Any annotation for the cell.
    string annotation;

    // Checks and sets the cell value.
    void CheckAndSet(Type t) {
      CHECK(type == t || type == NONE);
      type = t;
    }

    // Adds a numeric value to the cell's value.
    void Add(int64 value) {
      CheckAndSet(INT);
      int_value += value;
    }
    void Add(float value) {
      CheckAndSet(FLOAT);
      float_value += value;
    }

    // Sets the cell value.
    void Set(int64 value) {
      CheckAndSet(INT);
      int_value = value;
    }
    void Set(float value) {
      CheckAndSet(FLOAT);
      float_value = value;
    }
    void Set(const string &value) {
      CheckAndSet(STRING);
      str_value = value;
    }
  };

  // Represents a single table.
  struct Table {
    // Table name.
    string name;

    // Column name -> Column index.
    std::unordered_map<string, int> columns;

    // Row alias -> Row index. Not all rows will have an alias.
    std::unordered_map<string, int> rows;

    // Row number -> (Column number -> Cell).
    std::vector<std::vector<Cell>> cells;

    // Pretty printing options for the table.
    bool commas = true;
    int decimals = 2;

    Table() {}
    Table(const string &n) : name(n) {}

    // Makes a new cell or retrieves an existing one at (row, col).
    Cell *GetOrMakeCell(int row, int col);

    // Retrieves the row with alias 'row' or dies on failure.
    int GetRowOrDie(const string &row);

    // Retrieves the column with name 'col' or dies on failure.
    int GetColOrDie(const string &col);
  };

  // All tables created so far.
  std::vector<Table> tables_;

  // Current table.
  Table *current_ = nullptr;
};

}  // namespace sling

#endif  // UTIL_TABLE_WRITER_H_
