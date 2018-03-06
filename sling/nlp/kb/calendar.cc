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

Date::Date(const Object &object) {
  year_ = month_ = day_ = 0;
  precision_ = NONE;
  if (object.IsInt()) {
    int num = object.AsInt();
    CHECK(num > 0);
    if (num >= 1000000) {
      // YYYYMMDD
      year_ = num / 10000;
      month_ = (num % 10000) / 100;
      day_ = num % 100;
      precision_ = DAY;
    } else if (num >= 10000) {
      // YYYYMM
      year_ = num / 100;
      month_ = num % 100;
      precision_ = MONTH;
    } else if (num >= 1000) {
      // YYYY
      year_ = num;
      precision_ = YEAR;
    } else if (num >= 100) {
      // YYY*
      year_ = num * 10;
      precision_ = DECADE;
    } else if (num >= 10) {
      // YY**
      year_ = num * 100 + 1;
      precision_ = CENTURY;
    } else {
      // Y***
      year_ = num * 1000 + 1;
      precision_ = MILLENNIUM;
    }
  } else if (object.IsString()) {
    // Parse string date format: [+-]YYYY-MM-DDT00:00:00Z.
    Text datestr = object.AsString().text();
    const char *p = datestr.data();
    const char *end = p + datestr.size();

    // Parse + and - for AD and BC.
    bool bc = false;
    if (p < end && *p == '+') {
      p++;
    } else if (p < end && *p == '-') {
      bc = true;
      p++;
    }

    // Parse year, which can have trailing *s to indicate precision.
    p = parse_number(p, end, &year_);
    int stars = 0;
    while (p < end && *p == '*') {
      p++;
      stars++;
    }
    switch (stars) {
      case 0: precision_ = YEAR; break;
      case 1: precision_ = DECADE; year_ = year_ * 10; break;
      case 2: precision_ = CENTURY; year_ = year_ * 100 + 1; break;
      case 3: precision_ = MILLENNIUM; year_ = year_ * 1000 + 1; break;
    }
    if (bc) year_ = -year_;

    // Parse day and month.
    if (p < end && *p == '-') {
      p++;
      p = parse_number(p, end, &month_);
      if (month_ != 0) precision_ = MONTH;
      if (p < end && *p == '-') {
        p++;
        p = parse_number(p, end, &day_);
        if (day_ != 0) precision_ = DAY;
      }
    }
  }
}

void Calendar::Init(Store *store) {
  // Get symbols.
  store_ = store;
  n_name_ = store->Lookup("name");

  // Get calendar from store.
  Frame cal(store, "/w/calendar");
  if (!cal.valid()) return;

  // Get weekdays.
  for (const Slot &s : cal.GetFrame("/w/weekdays")) {
    weekdays_[s.name.AsInt()] = s.value;
  }

  // Get months.
  for (const Slot &s : cal.GetFrame("/w/months")) {
    months_[s.name.AsInt()] = s.value;
  }

  // Get days.
  for (const Slot &s : cal.GetFrame("/w/days")) {
    days_[s.name.AsInt()] = s.value;
  }

  // Get years.
  for (const Slot &s : cal.GetFrame("/w/years")) {
    years_[s.name.AsInt()] = s.value;
  }

  // Get decades.
  for (const Slot &s : cal.GetFrame("/w/decades")) {
    decades_[s.name.AsInt()] = s.value;
  }

  // Get centuries.
  for (const Slot &s : cal.GetFrame("/w/centuries")) {
    centuries_[s.name.AsInt()] = s.value;
  }

  // Get millennia.
  for (const Slot &s : cal.GetFrame("/w/millennia")) {
    millennia_[s.name.AsInt()] = s.value;
  }
};

string Calendar::DateAsString(const Date &date) const {
  // Parse date.
  Text year = YearName(date.year());

  switch (date.precision()) {
    case Date::NONE:
      return "";

    case Date::MILLENNIUM: {
      Text millennium = MillenniumName(date.year());
      if (!millennium.empty()) {
        return millennium.str();
      } else if (date.year() > 0) {
        return StrCat((date.year() - 1) / 1000 + 1, ". millennium AD");
      } else {
        return StrCat(-((date.year() + 1) / 1000 - 1), ". millennium BC");
      }
    }

    case Date::CENTURY: {
      Text century = CenturyName(date.year());
      if (!century.empty()) {
        return century.str();
      } else if (date.year() > 0) {
        return StrCat((date.year() - 1) / 100 + 1, ". century AD");
      } else {
        return StrCat(-((date.year() + 1) / 100 - 1), ". century BC");
      }
    }

    case Date::DECADE: {
      Text decade = DecadeName(date.year());
      if (!decade.empty()) {
        return decade.str();
      } else {
        return StrCat(year, "s");
      }
    }

    case Date::YEAR: {
      if (!year.empty()) {
        return year.str();
      } else if (date.year() > 0) {
        return StrCat(date.year());
      } else {
        return StrCat(-date.year(), " BC");
      }
    }

    case Date::MONTH: {
      Text month = MonthName(date.month());
      if (!month.empty()) {
        if (!year.empty()) {
          return StrCat(month, " ", year);
        } else {
          return StrCat(month, " ", date.year());
        }
      } else {
        return StrCat(date.year(), "-", date.month());
      }
    }

    case Date::DAY: {
      Text day = DayName(date.month(), date.day());
      if (!day.empty()) {
        if (!year.empty()) {
          return StrCat(day, ", ", year);
        } else {
          return StrCat(day, ", ", date.year());
        }
      } else {
        return StrCat(date.year(), "-", date.month(), "-", date.day());
      }
    }
  }

  return "???";
}

Handle Calendar::Day(const Date &date) const {
  if (date.precision() < Date::DAY) return Handle::nil();
  return Day(date.month(), date.day());
}

Handle Calendar::Day(int month, int day) const {
  auto f = days_.find(month * 100 + day);
  return f != days_.end() ? f->second : Handle::nil();
}

Handle Calendar::Month(const Date &date) const {
  if (date.precision() < Date::MONTH) return Handle::nil();
  return Month(date.month());
}

Handle Calendar::Month(int month) const {
  auto f = months_.find(month);
  return f != months_.end() ? f->second : Handle::nil();
}

Handle Calendar::Year(const Date &date) const {
  if (date.precision() < Date::YEAR) return Handle::nil();
  return Year(date.year());
}

Handle Calendar::Year(int year) const {
  auto f = years_.find(year);
  return f != years_.end() ? f->second : Handle::nil();
}

Handle Calendar::Decade(const Date &date) const {
  if (date.precision() < Date::DECADE) return Handle::nil();
  return Decade(date.year());
}

Handle Calendar::Decade(int year) const {
  int decade = year / 10;
  if (year < 0) decade--;
  auto f = decades_.find(decade);
  return f != decades_.end() ? f->second : Handle::nil();
}

Handle Calendar::Century(const Date &date) const {
  if (date.precision() < Date::CENTURY) return Handle::nil();
  return Century(date.year());
}

Handle Calendar::Century(int year) const {
  int century = year > 0 ? (year - 1) / 100 + 1 : (year + 1) / 100 - 1;
  auto f = centuries_.find(century);
  return f != centuries_.end() ? f->second : Handle::nil();
}

Handle Calendar::Millennium(const Date &date) const {
  if (date.precision() < Date::MILLENNIUM) return Handle::nil();
  return Millennium(date.year());
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
  int year = date.year();
  if (year < 1000 || year > 9999) return -1;
  switch (date.precision()) {
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
      return date.year() * 100 + date.month();
    case Date::DAY:
      return date.year() * 10000 + date.month() * 100 + date.day();
  }

  return -1;
}

string Calendar::DateString(const Date &date) {
  char str[16];
  *str = 0;
  int year = date.year();
  if (year >= -9999 && year <= 9999 && year != 0) {
    switch (date.precision()) {
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
        sprintf(str, "%+03d*", year / 10);
        break;
      case Date::YEAR:
        sprintf(str, "%+04d", year);
        break;
      case Date::MONTH:
        sprintf(str, "%+04d-%02d", date.year(), date.month());
        break;
      case Date::DAY:
        sprintf(str, "%+04d-%02d-%02d", date.year(), date.month(), date.day());
        break;
    }
  }

  return str;
}

}  // namespace nlp
}  // namespace sling

