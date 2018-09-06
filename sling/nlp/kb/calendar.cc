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

#include "sling/nlp/kb/calendar.h"

#include <stdio.h>

#include "sling/base/logging.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/string/strcat.h"
#include "sling/string/text.h"

namespace sling {
namespace nlp {

static const char *parse_number(const char *p, const char *end, int *value) {
  int n = 0;
  while (p < end && *p >= '0' && *p <= '9') {
    n = n * 10 + (*p++ - '0');
  }
  *value = n;
  return p;
}

void Date::Init(const Object &object) {
  year = month = day = 0;
  precision = NONE;
  if (object.IsInt()) {
    int num = object.AsInt();
    CHECK(num >= 0);
    ParseFromNumber(num);
  } else if (object.IsString()) {
    Text datestr = object.AsString().text();
    ParseFromString(datestr);
  } else if (object.IsFrame()) {
    Frame frame = object.AsFrame();
    ParseFromFrame(frame);
  }
}

void Date::ParseFromNumber(int num) {
  if (num >= 1000000) {
    // YYYYMMDD
    year = num / 10000;
    month = (num % 10000) / 100;
    day = num % 100;
    precision = DAY;
  } else if (num >= 10000) {
    // YYYYMM
    year = num / 100;
    month = num % 100;
    precision = MONTH;
  } else if (num >= 1000) {
    // YYYY
    year = num;
    precision = YEAR;
  } else if (num >= 100) {
    // YYY*
    year = num * 10;
    precision = DECADE;
  } else if (num >= 10) {
    // YY**
    year = num * 100 + 1;
    precision = CENTURY;
  } else if (num >= 0) {
    // Y***
    year = num * 1000 + 1;
    precision = MILLENNIUM;
  }
}

void Date::ParseFromString(Text str) {
  const char *p = str.data();
  const char *end = p + str.size();

  // Parse + and - for AD and BC.
  bool bc = false;
  if (p < end && *p == '+') {
    p++;
  } else if (p < end && *p == '-') {
    bc = true;
    p++;
  }

  // Parse year, which can have trailing *s to indicate precision.
  p = parse_number(p, end, &year);
  int stars = 0;
  while (p < end && *p == '*') {
    p++;
    stars++;
  }
  switch (stars) {
    case 0: precision = YEAR; break;
    case 1: precision = DECADE; year = year * 10; break;
    case 2: precision = CENTURY; year = year * 100 + 1; break;
    case 3: precision = MILLENNIUM; year = year * 1000 + 1; break;
  }
  if (bc) year = -year;

  // Parse day and month.
  if (p < end && *p == '-') {
    p++;
    p = parse_number(p, end, &month);
    if (month != 0) precision = MONTH;
    if (p < end && *p == '-') {
      p++;
      p = parse_number(p, end, &day);
      if (day != 0) precision = DAY;
    }
  }
}

void Date::ParseFromFrame(const Frame &frame) {
  // Try to get the 'point in time' property from frame and parse it.
  Store *store = frame.store();
  Object time(store, store->Resolve(frame.GetHandle("P585")));
  if (time.invalid()) return;
  if (time.IsInt()) {
    int num = time.AsInt();
    CHECK(num > 0);
    ParseFromNumber(num);
  } else if (time.IsString()) {
    Text datestr = time.AsString().text();
    ParseFromString(datestr);
  }
}

string Date::ISO8601() const {
  char str[32];
  *str = 0;
  switch (precision) {
    case Date::NONE: break;
    case Date::MILLENNIUM:
    case Date::CENTURY:
    case Date::DECADE:
    case Date::YEAR:
      sprintf(str, "%+05d-00-00T00:00:00Z", year);
      break;
    case Date::MONTH:
      sprintf(str, "%+05d-%02d-00T00:00:00Z", year, month);
      break;
    case Date::DAY:
      sprintf(str, "%+05d-%02d-%02dT00:00:00Z", year, month, day);
      break;
  }

  return str;
}

void Calendar::Init(Store *store) {
  // Get symbols.
  store_ = store;
  n_name_ = store->Lookup("name");

  // Get calendar from store.
  Frame cal(store, "/w/calendar");
  if (!cal.valid()) return;

  // Build calendar mappings.
  BuildCalendarMapping(&weekdays_, cal.GetFrame("/w/weekdays"));
  BuildCalendarMapping(&months_, cal.GetFrame("/w/months"));
  BuildCalendarMapping(&days_, cal.GetFrame("/w/days"));
  BuildCalendarMapping(&years_, cal.GetFrame("/w/years"));
  BuildCalendarMapping(&decades_, cal.GetFrame("/w/decades"));
  BuildCalendarMapping(&centuries_, cal.GetFrame("/w/centuries"));
  BuildCalendarMapping(&millennia_, cal.GetFrame("/w/millennia"));
};

bool Calendar::BuildCalendarMapping(CalendarMap *mapping, const Frame &source) {
  if (!source.valid()) return false;
  for (const Slot &s : source) {
    (*mapping)[s.name.AsInt()] = s.value;
  }
  return true;
}

string Calendar::DateAsString(const Date &date) const {
  // Parse date.
  Text year = YearName(date.year);

  switch (date.precision) {
    case Date::NONE:
      return "";

    case Date::MILLENNIUM: {
      Text millennium = MillenniumName(date.year);
      if (!millennium.empty()) {
        return millennium.str();
      } else if (date.year > 0) {
        return StrCat((date.year - 1) / 1000 + 1, ". millennium AD");
      } else {
        return StrCat(-((date.year + 1) / 1000 - 1), ". millennium BC");
      }
    }

    case Date::CENTURY: {
      Text century = CenturyName(date.year);
      if (!century.empty()) {
        return century.str();
      } else if (date.year > 0) {
        return StrCat((date.year - 1) / 100 + 1, ". century AD");
      } else {
        return StrCat(-((date.year + 1) / 100 - 1), ". century BC");
      }
    }

    case Date::DECADE: {
      Text decade = DecadeName(date.year);
      if (!decade.empty()) {
        return decade.str();
      } else {
        return StrCat(year, "s");
      }
    }

    case Date::YEAR: {
      if (!year.empty()) {
        return year.str();
      } else if (date.year > 0) {
        return StrCat(date.year);
      } else {
        return StrCat(-date.year, " BC");
      }
    }

    case Date::MONTH: {
      Text month = MonthName(date.month);
      if (!month.empty()) {
        if (!year.empty()) {
          return StrCat(month, " ", year);
        } else {
          return StrCat(month, " ", date.year);
        }
      } else {
        return StrCat(date.year, "-", date.month);
      }
    }

    case Date::DAY: {
      Text day = DayName(date.month, date.day);
      if (!day.empty()) {
        if (!year.empty()) {
          return StrCat(day, ", ", year);
        } else {
          return StrCat(day, ", ", date.year);
        }
      } else {
        return StrCat(date.year, "-", date.month, "-", date.day);
      }
    }
  }

  return "???";
}

Handle Calendar::Day(const Date &date) const {
  if (date.precision < Date::DAY) return Handle::nil();
  return Day(date.month, date.day);
}

Handle Calendar::Day(int month, int day) const {
  auto f = days_.find(month * 100 + day);
  return f != days_.end() ? f->second : Handle::nil();
}

Handle Calendar::Month(const Date &date) const {
  if (date.precision < Date::MONTH) return Handle::nil();
  return Month(date.month);
}

Handle Calendar::Month(int month) const {
  auto f = months_.find(month);
  return f != months_.end() ? f->second : Handle::nil();
}

Handle Calendar::Year(const Date &date) const {
  if (date.precision < Date::YEAR) return Handle::nil();
  return Year(date.year);
}

Handle Calendar::Year(int year) const {
  auto f = years_.find(year);
  return f != years_.end() ? f->second : Handle::nil();
}

Handle Calendar::Decade(const Date &date) const {
  if (date.precision < Date::DECADE) return Handle::nil();
  return Decade(date.year);
}

Handle Calendar::Decade(int year) const {
  int decade = year / 10;
  if (year < 0) decade--;
  auto f = decades_.find(decade);
  return f != decades_.end() ? f->second : Handle::nil();
}

Handle Calendar::Century(const Date &date) const {
  if (date.precision < Date::CENTURY) return Handle::nil();
  return Century(date.year);
}

Handle Calendar::Century(int year) const {
  int century = year > 0 ? (year - 1) / 100 + 1 : (year + 1) / 100 - 1;
  auto f = centuries_.find(century);
  return f != centuries_.end() ? f->second : Handle::nil();
}

Handle Calendar::Millennium(const Date &date) const {
  if (date.precision < Date::MILLENNIUM) return Handle::nil();
  return Millennium(date.year);
}

Handle Calendar::Millennium(int year) const {
  int millennium = year > 0 ? (year - 1) / 1000 + 1 : (year + 1) / 1000 - 1;
  auto f = millennia_.find(millennium);
  return f != millennia_.end() ? f->second : Handle::nil();
}

Text Calendar::ItemName(Handle item) const {
  if (!store_->IsFrame(item)) return "";
  FrameDatum *frame = store_->GetFrame(item);
  Handle name = frame->get(n_name_);
  if (!store_->IsString(name)) return "";
  StringDatum *str = store_->GetString(name);
  return str->str();
}

int Calendar::DateNumber(const Date &date) {
  int year = date.year;
  if (year < 1000 || year > 9999) return -1;
  switch (date.precision) {
    case Date::NONE:
      return -1;
    case Date::MILLENNIUM: {
      int mille = year > 0 ? (year - 1) / 1000 : (year + 1) / 1000;
      if (mille < 0) return -1;
      return mille;
    }
    case Date::CENTURY: {
      int cent = (year - 1) / 100;
      if (cent < 10) return -1;
      return cent;
    }
    case Date::DECADE:
      return year / 10;
    case Date::YEAR:
      return year;
    case Date::MONTH:
      return date.year * 100 + date.month;
    case Date::DAY:
      return date.year * 10000 + date.month * 100 + date.day;
  }

  return -1;
}

string Calendar::DateString(const Date &date) {
  char str[16];
  *str = 0;
  int year = date.year;
  if (year >= -9999 && year <= 9999 && year != 0) {
    switch (date.precision) {
      case Date::NONE: break;
      case Date::MILLENNIUM:
        if (year > 0) {
          int mille = (year - 1) / 1000;
          CHECK_GE(mille, 0) << year;
          CHECK_LE(mille, 9) << year;
          sprintf(str, "+%01d***", mille);
        } else {
          int mille = (year + 1) / -1000;
          CHECK_GE(mille, 0) << year;
          CHECK_LE(mille, 9) << year;
          sprintf(str, "-%01d***", mille);
        }
        break;
      case Date::CENTURY:
        if (year > 0) {
          int cent = (year - 1) / 100;
          CHECK_GE(cent, 0) << year;
          CHECK_LE(cent, 99) << year;
          sprintf(str, "+%02d**", cent);
        } else {
          int cent = (year + 1) / -100;
          CHECK_GE(cent, 0) << year;
          CHECK_LE(cent, 99) << year;
          sprintf(str, "-%02d**", cent);
        }
        break;
      case Date::DECADE:
        sprintf(str, "%+04d*", year / 10);
        break;
      case Date::YEAR:
        sprintf(str, "%+05d", year);
        break;
      case Date::MONTH:
        sprintf(str, "%+05d-%02d", date.year, date.month);
        break;
      case Date::DAY:
        sprintf(str, "%+05d-%02d-%02d", date.year, date.month, date.day);
        break;
    }
  }

  return str;
}

}  // namespace nlp
}  // namespace sling

