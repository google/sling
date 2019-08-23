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

#include "sling/base/logging.h"
#include "sling/frame/object.h"
#include "sling/frame/store.h"
#include "sling/string/strcat.h"
#include "sling/string/text.h"
#include "sling/util/unicode.h"

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
  if (frame.invalid()) return;
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

int Date::AsNumber() const {
  if (year < 1000 || year > 9999) return -1;
  switch (precision) {
    case NONE:
      return -1;
    case MILLENNIUM: {
      int mille = year > 0 ? (year - 1) / 1000 : (year + 1) / 1000;
      if (mille < 0) return -1;
      return mille;
    }
    case CENTURY: {
      int cent = (year - 1) / 100;
      if (cent < 10) return -1;
      return cent;
    }
    case DECADE:
      return year / 10;
    case YEAR:
      return year;
    case MONTH:
      return year * 100 + month;
    case DAY:
      return year * 10000 + month * 100 + day;
  }

  return -1;
}

Handle Date::AsHandle(Store *store) const {
  int number = AsNumber();
  if (number != -1) return Handle::Integer(number);
  string ts = AsString();
  if (!ts.empty()) return store->AllocateString(ts);
  return Handle::nil();
}

string Date::AsString() const {
  char str[16];
  *str = 0;
  if (year >= -9999 && year <= 9999 && year != 0) {
    switch (precision) {
      case NONE: break;
      case MILLENNIUM:
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
      case CENTURY:
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
      case DECADE:
        sprintf(str, "%+04d*", year / 10);
        break;
      case YEAR:
        sprintf(str, "%+05d", year);
        break;
      case MONTH:
        sprintf(str, "%+05d-%02d", year, month);
        break;
      case DAY:
        sprintf(str, "%+05d-%02d-%02d", year, month, day);
        break;
    }
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
  BuildCalendarMapping(&weekdays_, nullptr, cal.GetFrame("/w/weekdays"));
  BuildCalendarMapping(&months_, &month_items_, cal.GetFrame("/w/months"));
  BuildCalendarMapping(&days_, &day_items_, cal.GetFrame("/w/days"));
  BuildCalendarMapping(&years_, nullptr, cal.GetFrame("/w/years"));
  BuildCalendarMapping(&decades_, nullptr, cal.GetFrame("/w/decades"));
  BuildCalendarMapping(&centuries_, nullptr, cal.GetFrame("/w/centuries"));
  BuildCalendarMapping(&millennia_, nullptr, cal.GetFrame("/w/millennia"));
};

bool Calendar::BuildCalendarMapping(CalendarMap *mapping,
                                    CalendarItemMap *items,
                                    const Frame &source) {
  if (!source.valid()) return false;
  for (const Slot &s : source) {
    (*mapping)[s.name.AsInt()] = s.value;
    if (items != nullptr) {
      (*items)[s.value] = s.name.AsInt();
    }
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

bool Calendar::GetDayAndMonth(Handle item, Date *date) const {
  auto f = day_items_.find(item);
  if (f == day_items_.end()) return false;
  date->day = f->second % 100;
  date->month = f->second / 100;
  return true;
}

bool Calendar::GetMonth(Handle item, Date *date) const {
  auto f = month_items_.find(item);
  if (f == month_items_.end()) return false;
  date->month = f->second;
  return true;
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

void DateFormat::Init(const Frame &format) {
  // Get month names.
  Store *store = format.store();
  int prefix = format.GetInt("/w/month_abbrev");
  Array months = format.Get("/w/month_names").AsArray();
  if (months.valid()) {
    for (int i = 0; i < months.length(); ++i) {
      String month(store, months.get(i));
      CHECK(month.valid());
      string name = month.value();
      month_names_.push_back(name);

      // Normalize string.
      string lcname;
      month_dictionary_[UTF8::Lower(name)] = i + 1;
      if (prefix > 0) {
        string abbrev = UTF8::Lower(name.substr(0, prefix));
        month_dictionary_[abbrev] = i + 1;
      }
    }
  }

  // Get numeric and textual input date formats.
  Handle n_numeric = store->Lookup("/w/numeric_date_format");
  Handle n_textual = store->Lookup("/w/text_date_format");
  for (const Slot &s : format) {
    if (s.name == n_numeric) {
      String fmt(store, s.value);
      CHECK(fmt.valid());
      numeric_formats_.push_back(fmt.value());
    } else if (s.name == n_textual) {
      String fmt(store, s.value);
      CHECK(fmt.valid());
      text_formats_.push_back(fmt.value());
    }
  }

  // Get output format for dates.
  day_format_ = format.GetString("/w/day_output_format");
  month_format_ = format.GetString("/w/month_output_format");
  year_format_ = format.GetString("/w/year_output_format");
}

bool DateFormat::Parse(Text str, Date *date) const {
  // Determine if date is numeric or text.
  bool numeric = true;
  for (char c : str) {
    if (!IsDigit(c) && !IsDelimiter(c)) {
      numeric = false;
      break;
    }
  }

  // Parse date into year, month, and date component.
  bool valid = false;
  int len = str.size();
  int y, m, d, ys;
  if (numeric) {
    // Try to parse numeric date using each of the numeric date formats.
    for (const string &fmt : numeric_formats_) {
      if (len != fmt.size()) continue;
      y = m = d = ys = 0;
      valid = true;
      for (int i = 0; i < fmt.size(); ++i) {
        char c = str[i];
        char f = fmt[i];
        if (IsDigit(c)) {
          switch (f) {
            case 'Y': y = y * 10 + Digit(c); ys++; break;
            case 'M': m = m * 10 + Digit(c); break;
            case 'D': d = d * 10 + Digit(c); break;
            default: valid = false;
          }
        } else if (c != f) {
          valid = false;
        }
        if (!valid) break;
      }
      if (valid) break;
    }
  } else {
    // Try to parse text date format.
    for (const string &fmt : text_formats_) {
      y = m = d = ys = 0;
      valid = true;
      int j = 0;
      for (int i = 0; i < fmt.size(); ++i) {
        if (j >= len) valid = false;
        if (!valid) break;

        switch (fmt[i]) {
          case 'Y':
            // Parse sequence of digits as year.
            if (IsDigit(str[j])) {
              while (j < len && IsDigit(str[j])) {
                y = y * 10 + Digit(str[j++]);
                ys++;
              }
            } else {
              valid = false;
            }
            break;

          case 'M': {
            // Parse next token as month name.
            int k = j;
            while (k < len && !IsMonthBreak(str[k])) k++;
            int month = Month(Text(str, j, k - j));
            if (month != -1) {
              m = month;
              j = k;
            } else {
              valid = false;
            }
            break;
          }

          case 'D':
            // Parse sequence of digits as day.
            if (IsDigit(str[j])) {
              while (j < len && IsDigit(str[j])) {
                d = d * 10 + Digit(str[j++]);
              }
            } else {
              valid = false;
            }
            break;

          case ' ':
            // Skip sequence of white space.
            if (str[j] != ' ') {
              valid = false;
            } else {
              while (j < len && str[j] == ' ') j++;
            }
            break;

          default:
            // Literal match.
            valid = (fmt[i] == str[j++]);
        }
      }
      if (valid) break;
    }
  }

  // Return parsed date if it is valid.
  if (valid) {
    // Dates with two-digit years will have the years from 1970 to 2069.
    if (ys == 2) {
      if (y < 70) {
        y += 2000;
      } else {
        y += 1900;
      }
    }
    date->year = y;
    date->month = m;
    date->day = d;

    // Determine precision.
    if (date->year != 0) {
      date->precision = Date::YEAR;
      if (date->month != 0) {
        date->precision = Date::MONTH;
        if (date->day != 0) {
          date->precision = Date::DAY;
        }
      }
    }
  }
  return valid;
}

string DateFormat::AsString(const Date &date) const {
  Text format;
  switch (date.precision) {
    case Date::YEAR: format = year_format_; break;
    case Date::MONTH: format = month_format_; break;
    case Date::DAY: format = day_format_; break;
    default: return date.AsString();
  }
  string str;
  char buf[32];
  for (char c : format) {
    switch (c) {
      case 'Y':
        if (date.year != 0) {
          sprintf(buf, "%04d", date.year);
          str.append(buf);
        }
        break;

      case 'M':
        if (date.month != 0) {
          if (date.month > 0 && date.month <= month_names_.size()) {
            str.append(month_names_[date.month - 1]);
          } else {
            sprintf(buf, "?%02d?", date.month);
            str.append(buf);
          }
        }
        break;

      case 'D':
        if (date.day != 0) {
          sprintf(buf, "%d", date.day);
          str.append(buf);
        }
        break;

      default:
        str.push_back(c);
    }
  }
  return str;
}

int DateFormat::Month(Text name) const {
  string lower;
  UTF8::Lowercase(name.data(), name.size(), &lower);
  auto f = month_dictionary_.find(lower);
  if (f == month_dictionary_.end()) return -1;
  return f->second;
}

}  // namespace nlp
}  // namespace sling

