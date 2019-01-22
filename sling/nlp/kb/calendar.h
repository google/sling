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

#ifndef SLING_NLP_KB_CALENDAR_H_
#define SLING_NLP_KB_CALENDAR_H_

#include <string>
#include <unordered_map>

#include "sling/base/types.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/string/text.h"

namespace sling {
namespace nlp {

// Date with day, month, and year.
class Date {
 public:
  // Granularity for date.
  enum Precision {NONE, MILLENNIUM, CENTURY, DECADE, YEAR, MONTH, DAY};

  // Initialize invalid date.
  Date() {}

  // Initialize date from parts.
  Date(int year, int month, int day, Precision precision)
      : year(year), month(month), day(day), precision(precision) {}

  // Initialize date from object.
  Date(const Object &object) { Init(object); }
  void Init(const Object &object);

  // Parse date from number.
  void ParseFromNumber(int num);

  // Parse string date format: [+-]YYYY-MM-DDT00:00:00Z.
  void ParseFromString(Text str);

  // Parse date from frame.
  void ParseFromFrame(const Frame &frame);

  // Date in  ISO 8601 format, e.g. "+2013-05-01T00:00:00Z" is May 1, 2013.
  string ISO8601() const;

  // Convert date to integer or return -1 if the date cannot be encoded as an
  // integer. This can only be used for dates after 1000 AD.
  int AsNumber() const;

  // Return an integer or string handle representing date.
  Handle AsHandle(Store *store) const;

  // Convert date to string format. The date format depends on the precision:
  // DAY:        [+|-]YYYY-MM-DD
  // MONTH:      [+|-]YYYY-MM
  // YEAR:       [+|-]YYYY
  // DECADE:     [+|-]YYY*
  // CENTURY:    [+|-]YY**
  // MILLENNIUM: [+|-]Y***
  string AsString() const;

  // Year or 0 if date is invalid.
  int year = 0;

  /// Month (1=January) or 0 if no month in date.
  int month = 0;

  // Day of month (first day of month is 1) or 0 if no day in date.
  int day = 0;

  // Date precision.
  Precision precision = NONE;
};

// Calendar with items about (Gregorian) calendar concepts.
class Calendar {
 public:
  // Initialize calendar from store.
  void Init(Store *store);

  // Convert date to human-readable string.
  string DateAsString(const Date &date) const;
  string DateAsString(const Object &object) const {
    return DateAsString(Date(object));
  }

  // Get day and month for calendar item.
  bool GetDayAndMonth(Handle item, Date *date) const;

  // Get month for item.
  bool GetMonth(Handle item, Date *date) const;

  // Get item for day.
  Handle Day(const Date &date) const;
  Handle Day(int month, int day) const;
  Text DayName(int month, int day) const { return ItemName(Day(month, day)); }

  // Get item for month.
  Handle Month(const Date &date) const;
  Handle Month(int month) const;
  Text MonthName(int month) const { return ItemName(Month(month)); }

  // Get item for year.
  Handle Year(const Date &date) const;
  Handle Year(int year) const;
  Text YearName(int year) const { return ItemName(Year(year)); }

  // Get item for decade.
  Handle Decade(const Date &date) const;
  Handle Decade(int year) const;
  Text DecadeName(int year) const { return ItemName(Decade(year)); }

  // Get item for century.
  Handle Century(const Date &date) const;
  Handle Century(int year) const;
  Text CenturyName(int year) const { return ItemName(Century(year)); }

  // Get item for millennium.
  Handle Millennium(const Date &date) const;
  Handle Millennium(int year) const;
  Text MillenniumName(int year) const { return ItemName(Millennium(year)); }

 private:
  // Mapping from calendar item key to the corresponding calendar item.
  typedef std::unordered_map<int, Handle> CalendarMap;

  // Mapping from calendar item to the corresponding key.
  typedef HandleMap<int> CalendarItemMap;

  // Get name for item.
  Text ItemName(Handle item) const;

  // Build calendar mapping.
  bool BuildCalendarMapping(CalendarMap *mapping,
                            CalendarItemMap *items,
                            const Frame &source);

  // Store with calendar.
  Store *store_ = nullptr;

  // Symbols.
  Handle n_name_;

  // Weekdays (0=Sunday).
  CalendarMap weekdays_;

  // Months (1=January).
  CalendarMap months_;

  // Days of year where the key is month*100+day.
  CalendarMap days_;

  // Years where BCE are represented by negative numbers. There is no year 0 in
  // the AD calendar although there is an item for this concept.
  CalendarMap years_;

  // Decades. The decades are numbered as year/10, e.g. the decade from
  // 1970-1979 has 197 as the key. The last decade before AD, i.e. years 1-9 BC,
  // has decade number -1. Likewise, the first decade after AD, i.e. years
  // 1-9 AD, has number 0. These two decades only have nine years.
  CalendarMap decades_;

  // Centuries. The centuries are numbered as (year-1)/100+1 for AD and
  // (year+1)/100-1 for BC, so the 19. century, from 1801 to 1900, has number
  // 19. Centuries BCE are represented by negative numbers. The 1st century BC,
  // from 100BC to 1BC, has number -1.
  CalendarMap centuries_;

  // Millennia. The millennia are numbered as (year-1)/1000+1 for AD and
  // (year+1)/1000-1 for BC.
  CalendarMap millennia_;

  // Mapping from calendar item to day of year (month*100+day).
  CalendarItemMap day_items_;

  // Mapping from calendar item to month.
  CalendarItemMap month_items_;
};

// Date parser and generator based on language-dependent format, e.g.:
// {
//   /w/month_names: ["January", "February", ...]
//   /w/day_output_format: "D M Y"
//   /w/month_output_format: "M Y"
//   /w/year_output_format: "Y"
//   /w/numeric_date_format: "MM/DD/YYYY"
//   /w/numeric_date_format: "DD-MM-YYYY"
//   ...
//   /w/text_date_format: "M D, Y"
//   ...
// }
class DateFormat {
 public:
  // Initialize from date format configuration.
  void Init(const Frame &format);

  // Parse date.
  bool Parse(Text str, Date *date) const;

  // Convert date to string.
  string AsString(const Date &date) const;

  // Lookup month name. Return -1 for invalid month name.
  int Month(Text name) const;

 private:
  // Check if character is a digit.
  static bool IsDigit(char c) {
    return c >= '0' && c <= '9';
  }

  // Check if character is a date delimiter.
  static bool IsDelimiter(char c) {
    return c == '-' || c == '/' || c == '.';
  }

  // Check if character is a month name break.
  static bool IsMonthBreak(char c) {
    return c == ' ' || IsDigit(c) || IsDelimiter(c);
  }

  // Return digit value.
  static int Digit(char c) {
    DCHECK(IsDigit(c));
    return c - '0';
  }

  // Month names for generating dates.
  std::vector<string> month_names_;

  // Month names (and abbreviations) for parsing.
  std::unordered_map<string, int> month_dictionary_;

  // Numeric date input formats, e.g. 'YYYY-MM-DD'.
  std::vector<string> numeric_formats_;

  // Text date input formats, e.g. 'M D, Y'.
  std::vector<string> text_formats_;

  // Output formats for dates.
  string day_format_;
  string month_format_;
  string year_format_;
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_KB_CALENDAR_H_

