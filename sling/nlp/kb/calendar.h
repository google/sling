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

  // Convert date to integer or return -1 if the date cannot be encoded as an
  // integer. This can only be used for dates after 1000 AD.
  static int DateNumber(const Date &date);

  // Convert date to string format. The date format dependends on the precision:
  // DAY:        [+|-]YYYY-MM-DD
  // MONTH:      [+|-]YYYY-MM
  // YEAR:       [+|-]YYYY
  // DECADE:     [+|-]YYY*
  // CENTURY:    [+|-]YY**
  // MILLENNIUM: [+|-]Y***
  static string DateString(const Date &date);

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
  // Get name for item.
  Text ItemName(Handle item) const;

  // Mapping from calendar item key to the corresponding calendar item.
  typedef std::unordered_map<int, Handle> CalendarMap;

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
};

}  // namespace nlp
}  // namespace sling

#endif  // SLING_NLP_KB_CALENDAR_H_

