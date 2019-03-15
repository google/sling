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

#include "sling/nlp/wiki/wiki-annotator.h"

#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "sling/nlp/document/document.h"
#include "sling/nlp/document/fingerprinter.h"
#include "sling/nlp/document/lex.h"
#include "sling/nlp/document/document-tokenizer.h"
#include "sling/nlp/kb/calendar.h"
#include "sling/nlp/wiki/wiki.h"
#include "sling/string/numbers.h"
#include "sling/string/strcat.h"
#include "sling/string/text.h"
#include "sling/util/mutex.h"

namespace sling {
namespace nlp {

// Template macro that expands to a fixed text.
class TextTemplate : public WikiMacro {
 public:
  void Init(const Frame &config) override {
    text_ = config.GetString("text");
    link_ = config.GetHandle("link");
  }

  void Generate(const WikiTemplate &templ, WikiAnnotator *annotator) override {
    // Output text.
    int begin = annotator->position();
    annotator->Content(text_);
    int end = annotator->position();

    // Add link annotation.
    if (!link_.IsNil()) {
      annotator->AddMention(begin, end, link_);
    }
  }

 private:
  string text_;
  Handle link_;
};

REGISTER_WIKI_MACRO("text", TextTemplate);

// Template macro for expanding fractions.
class FracTemplate : public WikiMacro {
 public:
  void Generate(const WikiTemplate &templ, WikiAnnotator *annotator) override {
    switch (templ.NumArgs()) {
      case 0:
        annotator->Content("/");
        break;
      case 1:
        annotator->Content("1/");
        annotator->Content(templ.GetValue(1));
        break;
      case 2:
        annotator->Content(templ.GetValue(1));
        annotator->Content("/");
        annotator->Content(templ.GetValue(2));
        break;
      case 3:
        annotator->Content(templ.GetValue(1));
        annotator->Content("&nbsp;");
        annotator->Content(templ.GetValue(2));
        annotator->Content("/");
        annotator->Content(templ.GetValue(3));
        break;
    }
  }
};

REGISTER_WIKI_MACRO("frac", FracTemplate);

// Template macro for expanding arguments with open, close, and delimiter
// text.
class TagTemplate : public WikiMacro {
 public:
  void Init(const Frame &config) override {
    arg_ = config.GetString("arg");
    argnum_ = config.GetInt("argnum", 0);
    argname_ = config.GetString("argname");
    open_ = config.GetString("open");
    close_ = config.GetString("close");
    delimiter_ = config.GetString("delimiter");
  }

  void Generate(const WikiTemplate &templ, WikiAnnotator *annotator) override {
    annotator->Content(open_);
    if (argnum_ == -1 && argname_.empty()) {
      for (int i = 1; i < templ.NumArgs() + 1; ++i) {
        if (i != 1) annotator->Content(delimiter_);
        const Node *arg = templ.GetArgument(i);
        if (arg != nullptr) {
          templ.extractor()->ExtractNode(*arg);
        }
      }
    } else {
      const Node *content = templ.GetArgument(argname_, argnum_);
      if (content != nullptr) {
        templ.extractor()->ExtractNode(*content);
      }
    }
    annotator->Content(close_);
  }

 private:
  string arg_;
  int argnum_;
  string argname_;
  string open_;
  string close_;
  string delimiter_;
};

REGISTER_WIKI_MACRO("tag", TagTemplate);

// Template macro for defintions and abbreviations.
class DefineTemplate : public WikiMacro {
 public:
  void Generate(const WikiTemplate &templ, WikiAnnotator *annotator) override {
    const Node *abbreviation = templ.GetArgument(1);
    if (abbreviation != nullptr) {
      templ.extractor()->ExtractNode(*abbreviation);
    }
    const Node *meaning = templ.GetArgument(2);
    if (meaning != nullptr) {
      annotator->Content(" (");
      templ.extractor()->ExtractNode(*meaning);
      annotator->Content(")");
    }
  }
};

REGISTER_WIKI_MACRO("define", DefineTemplate);

// Template macro for expanding dates with annotations.
class DateTemplate : public WikiMacro {
 public:
  void Init(const Frame &config) override {
    // Get the date format from the configuration.
    Frame format = config.GetFrame("format");
    if (format.valid()) format_.Init(format);

    // Get date argument configuration.
    date_argnum_ = config.GetInt("full", -1);
    year_argnum_ = config.GetInt("year", -1);
    month_argnum_ = config.GetInt("month", -1);
    day_argnum_ = config.GetInt("day", -1);
    qualification_argnum_ = config.GetInt("qual", -1);
    reverse_args_ = config.GetBool("reverse");

    date_argname_ = config.GetString("fulln");
    year_argname_ = config.GetString("yearn");
    month_argname_ = config.GetString("monthn");
    day_argname_ = config.GetString("dayn");

    // Get prefix and postfix.
    prefix_ = config.GetString("pre");
    postfix_ = config.GetString("post");
  }

  void Generate(const WikiTemplate &templ, WikiAnnotator *annotator) override {
    // Parse input date.
    Date date;
    if (!GetDate(templ, &date)) {
      templ.Extract(1);
      return;
    }

    // Output date.
    annotator->Content(prefix_);
    if (date.precision != Date::NONE) {
      // Output formatted date.
      int begin = annotator->position();
      annotator->Content(format_.AsString(date));
      int end = annotator->position();

      // Create date annotation.
      Builder b(annotator->store());
      b.AddIsA("/w/time");
      b.AddIs(date.AsHandle(annotator->store()));
      annotator->AddMention(begin, end, b.Create().handle());
    } else {
      // Unable to parse date, output verbatim.
      templ.Extract(1);
    }
    if (qualification_argnum_ != -1) {
      templ.Extract(qualification_argnum_);
    }
    annotator->Content(postfix_);
  }

 protected:
  // Get date component.
  const Node *DateComponent(const WikiTemplate &templ, Text name, int index) {
    if (reverse_args_) {
      // In reverse argument mode you have {{date|y}}, {{date|m|y}}, or
      // {{date|d|m|y}}, compared to normal mode, where the order is
      // {{date|y}}, {{date|y|m}}, or {{date|y|m|d}}.
      int numargs = templ.NumArgs();
      if (numargs > 3) numargs = 3;
      index = numargs - index + 1;
    }
    return templ.GetArgument(name, index);
  }

  // Get date from template argument(s).
  bool GetDate(const WikiTemplate &templ, Date *date) {
    // Parse full date argument.
    const Node *full_arg = templ.GetArgument(date_argname_, date_argnum_);
    if (full_arg != nullptr) {
      string fulldate = templ.GetValue(full_arg);
      if (!format_.Parse(fulldate, date)) return false;
    }

    // Parse year argument.
    const Node *year_arg = DateComponent(templ, year_argname_, year_argnum_);
    if (year_arg != nullptr) {
      int y = templ.GetNumber(year_arg);
      if (y != -1) {
        date->year = y;
      } else {
        return false;
      }
    }

    // Parse month argument.
    const Node *month_arg = DateComponent(templ, month_argname_, month_argnum_);
    if (month_arg != nullptr) {
      int m = templ.GetNumber(month_arg);
      if (m == -1) {
        m = format_.Month(templ.GetValue(month_arg));
      }
      if (m != -1) {
        date->month = m;
      } else {
        return false;
      }
    }

    // Parse day argument.
    const Node *day_arg = DateComponent(templ, day_argname_, day_argnum_);
    if (day_arg != nullptr) {
      int d = templ.GetNumber(day_arg);
      if (d != -1) {
        date->day = d;
      } else {
        return false;
      }
    }

    // Check BCE.
    if (templ.GetArgument("BCE") || templ.GetArgument("BC")) {
      date->year = -date->year;
    }

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
    return true;
  }

 private:
  // Date format for language.
  DateFormat format_;

  // Date input argument.
  int date_argnum_ = -1;
  int year_argnum_ = -1;
  int month_argnum_ = -1;
  int day_argnum_ = -1;
  int qualification_argnum_ = -1;
  bool reverse_args_ = false;

  string date_argname_;
  string year_argname_;
  string month_argname_;
  string day_argname_;

  // Prefix and postfix text.
  string prefix_;
  string postfix_;
};

REGISTER_WIKI_MACRO("date", DateTemplate);

// Template macro for marriage.
class MarriageTemplate : public WikiMacro {
 public:
  void Init(const Frame &config) override {
    // Get the date format from the configuration.
    Frame format = config.GetFrame("format");
    if (format.valid()) format_.Init(format);
  }

  void Generate(const WikiTemplate &templ, WikiAnnotator *annotator) override {
    // Parse start date.
    Date married;
    Date ended;
    int numargs = templ.NumArgs();
    const Node *endarg = templ.GetArgument("end");
    if (numargs >= 2) format_.Parse(templ.GetValue(2), &married);
    if (numargs >= 3) format_.Parse(templ.GetValue(3), &ended);

    // Output spouse.
    templ.Extract(1);

    // Output marriage date.
    Handle marriage_start = Handle::nil();
    Handle marriage_end = Handle::nil();
    annotator->Content(" (");
    if (numargs >= 2) {
      annotator->Content("m. ");
      int begin = annotator->position();
      annotator->Content(format_.AsString(married));
      int end = annotator->position();

      // Create date annotation.
      if (married.precision != Date::NONE) {
        Builder b(annotator->store());
        b.AddIsA("/w/time");
        b.AddIs(married.AsHandle(annotator->store()));
        marriage_start = b.Create().handle();
        annotator->AddMention(begin, end, marriage_start);
      }
    }
    if (numargs >= 3) {
      if (endarg != nullptr) {
        annotator->Content("; ");
        templ.Extract(endarg);
        annotator->Content(" ");
      } else {
        annotator->Content(" &ndash; ");
      }
      int begin = annotator->position();
      annotator->Content(format_.AsString(ended));
      int end = annotator->position();

      // Create date annotation.
      if (ended.precision != Date::NONE) {
        Builder b(annotator->store());
        b.AddIsA("/w/time");
        b.AddIs(ended.AsHandle(annotator->store()));
        marriage_end = b.Create().handle();
        annotator->AddMention(begin, end, marriage_end);
      }
    }
    annotator->Content(")");

    // Add marriage thematic frame.
    Builder b(annotator->store());
    b.AddIsA("/wp/marriage");
    if (!marriage_start.IsNil()) b.Add("/wp/marriage/start", marriage_start);
    if (!marriage_end.IsNil()) b.Add("/wp/marriage/end", marriage_end);
    annotator->AddTheme(b.Create().handle());
  }

 private:
  // Date format for language.
  DateFormat format_;
};

REGISTER_WIKI_MACRO("marriage", MarriageTemplate);

// Template macro for years.
class YearTemplate : public WikiMacro {
 public:
  void Init(const Frame &config) override {
    bce_ = config.GetBool("bce");
    range_ = config.GetBool("range");
    bcarg_ = config.GetInt("bc");
    prefix_ = config.GetString("pre");
    postfix_ = config.GetString("post");
  }

  void Generate(const WikiTemplate &templ, WikiAnnotator *annotator) override {
    GenerateYear(templ, annotator, 1);
    if (range_ && templ.GetArgument(2)) {
      annotator->Content(" – ");
      GenerateYear(templ, annotator, 2);
    }
  }

  void GenerateYear(const WikiTemplate &templ, WikiAnnotator *annotator,
                    int argnum) {
    annotator->Content(prefix_);
    int year = templ.GetNumber(argnum);
    if (year != -1) {
      Date date(bce_ ? -year : year, 0, 0, Date::YEAR);
      string bc;
      if (bcarg_ != -1) {
        bc = templ.GetValue(bcarg_);
        if (!bc.empty()) date.year = -date.year;
      }

      int begin = annotator->position();
      templ.Extract(argnum);
      annotator->Content(bc);
      annotator->Content(postfix_);
      int end = annotator->position();

      Builder b(annotator->store());
      b.AddIsA("/w/time");
      b.AddIs(date.AsHandle(annotator->store()));
      annotator->AddMention(begin, end, b.Create().handle());
    } else {
      templ.Extract(argnum);
    }
  }

 private:
  bool bce_;
  bool range_;
  string prefix_;
  string postfix_;
  int bcarg_ = -1;
};

REGISTER_WIKI_MACRO("year", YearTemplate);

// Template macro for measures.
class MeasureTemplate : public WikiMacro {
 public:
  void Init(const Frame &config) override {
    Store *store = config.store();
    Frame units = config.GetFrame("units");
    if (units.valid()) {
      for (const Slot &s : units) {
        if (!store->IsString(s.name)) continue;
        String name(store, s.name);
        Frame info(store, s.value);
        CHECK(name.valid() && info.valid());
        Unit &unit = units_[name.value()];
        unit.item = info.GetHandle("/w/unit");
        unit.factor = info.GetFloat("/w/amount");
      }
    }
    value_argnum_ = config.GetInt("value", 1);
    unit_argnum_ = config.GetInt("unit", 2);
  }

  void Generate(const WikiTemplate &templ, WikiAnnotator *annotator) override {
    // Get value and unit.
    string value_text = templ.GetValue(value_argnum_);
    string unit_text = templ.GetValue(unit_argnum_);

    // Output measure.
    int begin = annotator->position();
    annotator->Content(value_text);
    annotator->Content(" ");
    annotator->Content(unit_text);
    int end = annotator->position();

    // Parse value.
    float amount;
    if (!safe_strtof(value_text, &amount)) return;

    // Look up unit.
    auto f = units_.find(unit_text);
    if (f == units_.end()) return;

    // Scale value.
    const Unit &unit = f->second;
    if (unit.factor != 0) amount *= unit.factor;

    // Add measure annotation.
    Builder b(annotator->store());
    b.AddIsA("/w/quantity");
    b.Add("/w/amount", amount);
    b.Add("/w/unit", unit.item);
    annotator->AddMention(begin, end, b.Create().handle());
  }

 private:
  struct Unit {
    Handle item;
    float factor;
  };

  // Units with scaling factors.
  std::unordered_map<string, Unit> units_;

  // Value and unit arguments.
  int value_argnum_;
  int unit_argnum_;
};

REGISTER_WIKI_MACRO("measure", MeasureTemplate);

// Template macro for info boxes.
class InfoboxTemplate : public WikiMacro {
 public:
  ~InfoboxTemplate() { if (docnames_) docnames_->Release(); }

  void Init(const Frame &config) override {
    Store *store = config.store();
    docnames_ = new DocumentNames(store);
    Handle n_class = store->Lookup("class");
    Handle n_fields = store->Lookup("fields");
    Handle n_group = store->Lookup("group");
    Handle n_alias = store->Lookup("alias");
    n_infobox_ = store->Lookup("/wp/infobox");
    for (const Slot &s : config) {
      if (s.name == n_class) {
        classes_.push_back(s.value);
      } else if (s.name == n_fields) {
        Frame fields(store, s.value);
        for (const Slot &f : fields) {
          if (!store->IsString(f.name)) continue;
          string name = String(store, f.name).value();
          Field &field = fields_[name];
          if (!field.key.IsNil()) {
            LOG(WARNING) << "Duplicate infobox field: " << name;
          }
          Frame key(store, f.value);
          if (key.IsAnonymous()) {
            field.key = key.GetHandle(Handle::is());
            field.group = key.GetHandle(n_group);
            field.alias = key.GetInt(n_alias, -1);
          } else {
            field.key = f.value;
            field.group = Handle::nil();
            field.alias = -1;
          }
        }
      }
    }
  }

  void Generate(const WikiTemplate &templ, WikiAnnotator *annotator) override {
    // Create builders for main fields and repeated fields.
    Store *store = annotator->store();
    Builder main(store);
    main.AddIsA(n_infobox_);
    for (Handle cls : classes_) main.AddIsA(cls);
    HandleMap<std::vector<Builder *>> groups;

    // Extract all the fields.
    std::vector<const Node *> arguments;
    templ.GetArguments(&arguments);
    for (const Node *arg : arguments) {
      // Skip empty fields.
      if (templ.IsEmpty(arg)) continue;

      // Look up field.
      string name = arg->name().str();
      auto f = fields_.find(name);
      Field *field = nullptr;
      int index = -1;
      if (f != fields_.end()) {
        field = &f->second;
        if (!field->group.IsNil()) index = 0;
      } else {
        // Try to remove number suffix for repeated field.
        int i = name.size() - 1;
        int power = 1;
        index = 0;
        while (i > 1 && name[i] >= '0' && name[i] <= '9') {
          index += (name[i--] - '0') * power;
          power *= 10;
        }
        if (index != 0) {
          name = name.substr(0, i + 1);
          auto f = fields_.find(name);
          if (f != fields_.end()) {
            field = &f->second;
          }
        }
      }

      if (field == nullptr) {
        templ.ExtractSkip(arg);
        VLOG(5) << "unknown field " << name;
        continue;
      }

      // Extract field using a sub-annotator.
      WikiAnnotator value(annotator);
      templ.extractor()->Enter(&value);
      templ.extractor()->ExtractNode(*arg);
      templ.extractor()->Leave(&value);

      // Convert field value to LEX format.
      Document document(store, docnames_);
      document.SetText(value.text());
      GetTokenizer()->Tokenize(&document);
      value.AddToDocument(&document);
      document.Update();
      string lex = ToLex(document);

      // Add field to infobox frame.
      if (field->group.IsNil()) {
        main.Add(field->key, lex);
      } else {
        auto &group = groups[field->group];
        if (group.size() < index + 1) group.resize(index + 1);
        Builder *&element = group[index];
        if (element == nullptr) element = new Builder(store);
        element->Add(field->key, lex);
      }

      // Extract alias.
      if (field->alias != -1) {
        // Extract alias text from field.
        AliasSink alias(field->alias);
        templ.extractor()->Enter(&alias);
        templ.extractor()->ExtractChildren(*arg);
        templ.extractor()->Leave(&alias);

        // Output aliases.
        std::istringstream aliases(alias.text());
        string name;
        while (getline(aliases, name)) {
          if (name.empty()) continue;
          AliasSource source = static_cast<AliasSource>(field->alias);
          annotator->AddAlias(name, source);
        }
      }
    }

    // Create frames for repeated fields and add them to the main frame.
    for (auto &it : groups) {
      for (Builder *element : it.second) {
        if (element != nullptr) {
          main.Add(it.first, element->Create().handle());
          delete element;
        }
      }
    }

    // Add infobox as thematic frame.
    annotator->AddTheme(main.Create().handle());
  }

 private:
  // Sink for collecting text from aliases.
  class AliasSink : public WikiSink {
   public:
    AliasSink(int type) : type_(type) {}

    void Content(const char *begin, const char *end) override {
      if (begin != end && *begin == '<') {
        line_break_ = true;
      } else {
        for (const char *p = begin; p < end; ++p) {
          if (*p == ' ') {
            space_break_ = true;
          } else if (*p == '\n') {
            line_break_ = true;
          } else if ((*p == ',' || *p == ';') &&
                     (line_break_ || type_ == SRC_WIKIPEDIA_NICKNAME)) {
            line_break_ = true;
          } else if (*p == '(' || *p == '[') {
            in_parentheses_ = true;
          } else if (*p == ')' || *p == ']') {
            in_parentheses_ = false;
          } else if (!in_parentheses_) {
            if (line_break_) {
              text_.push_back('\n');
              line_break_ = false;
              space_break_ = false;
            } else if (space_break_) {
              text_.push_back(' ');
              space_break_ = false;
            }
            text_.push_back(*p);
          }
        }
      }
    }

    // Font change starts a new alias.
    void Font(int font) override {
      line_break_ = true;
    }

    // Return extracted text.
    const string &text() const { return text_; }

   protected:
    // Extracted text.
    string text_;

    // Alias type.
    int type_;

    // Pending space and line breaks.
    bool space_break_ = false;
    bool line_break_ = false;

    // Parentheses tracking for skipping text inside parentheses.
    bool in_parentheses_ = false;
  };

  // Singleton document tokenizer.
  static DocumentTokenizer *GetTokenizer() {
    static Mutex mu;
    static DocumentTokenizer *tokenizer = nullptr;
    MutexLock lock(&mu);
    if (tokenizer == nullptr) tokenizer = new DocumentTokenizer();
    return tokenizer;
  }

  // Infobox field definition.
  struct Field {
    Handle key;     // slot name for field
    Handle group;   // group for repeated field
    int alias;      // alias type or -1 if it is not an alias field
  };

  // Types added to thematic frame for infobox.
  std::vector<Handle> classes_;

  // Infobox fields keyed by field name.
  std::unordered_map<string, Field> fields_;

  // Document names.
  DocumentNames *docnames_ = nullptr;

  // Infobox type.
  Handle n_infobox_;
};

REGISTER_WIKI_MACRO("infobox", InfoboxTemplate);

// Template macro for geographic coordinates (latitude and longitude).
class CoordinateTemplate : public WikiMacro {
 public:
  void Generate(const WikiTemplate &templ, WikiAnnotator *annotator) override {
    // Get latitude and longitude.
    float lat = 0.0;
    float lng = 0.0;
    int numargs = templ.NumArgs();
    switch (numargs) {
      case 2:
      case 3:
        lat = templ.GetFloat(1);
        lng = templ.GetFloat(2);
        break;

      case 4:
      case 5:
        lat = templ.GetFloat(1);
        if (templ.GetValue(2) == "S") lat = -lat;
        lng = templ.GetFloat(3);
        if (templ.GetValue(4) == "W") lng = -lng;
        break;

      case 6:
      case 7:
        lat = templ.GetFloat(1) + templ.GetFloat(2) / 60.0;
        if (templ.GetValue(3) == "S") lat = -lat;
        lng = templ.GetFloat(4) + templ.GetFloat(5) / 60.0;
        if (templ.GetValue(6) == "W") lng = -lng;
        break;

      case 8:
      case 9:
        lat = templ.GetFloat(1) +
              templ.GetFloat(2) / 60.0 +
              templ.GetFloat(3) / 3600.0;
        if (templ.GetValue(4) == "S") lat = -lat;
        lng = templ.GetFloat(5) +
              templ.GetFloat(6) / 60.0 +
              templ.GetFloat(7) / 3600.0;
        if (templ.GetValue(8) == "W") lng = -lng;
        break;
    }

    // Output coordinate.
    int begin = annotator->position();
    if (numargs == 8) {
      // Latitude.
      templ.Extract(1);
      annotator->Content("°");
      templ.Extract(2);
      annotator->Content("′");
      templ.Extract(3);
      annotator->Content("″");
      templ.Extract(4);

      // Longitude.
      templ.Extract(5);
      annotator->Content("°");
      templ.Extract(6);
      annotator->Content("′");
      templ.Extract(7);
      annotator->Content("″");
      templ.Extract(8);
    } else {
      annotator->Content(GeoCoord(lat, true));
      annotator->Content(" ");
      annotator->Content(GeoCoord(lng, false));
    }
    int end = annotator->position();

    // Create geo annotation.
    Builder b(annotator->store());
    b.AddIsA("/w/geo");
    b.Add("/w/lat", lat);
    b.Add("/w/lng", lng);
    annotator->AddMention(begin, end, b.Create().handle());
  }

 private:
  // Convert geo coordinate from decimal to minutes and seconds.
  static string GeoCoord(double coord, bool latitude) {
    // Compute direction.
    const char *sign;
    if (coord < 0) {
      coord = -coord;
      sign = latitude ? "S" : "W";
    } else {
      sign = latitude ? "N" : "E";
    }

    // Compute degrees.
    double integer;
    double remainder = modf(coord, &integer);
    int degrees = static_cast<int>(integer);

    // Compute minutes.
    remainder = modf(remainder * 60, &integer);
    int minutes = static_cast<int>(integer);

    // Compute seconds.
    remainder = modf(remainder * 60, &integer);
    int seconds = static_cast<int>(integer + 0.5);

    // Build coordinate string.
    return StrCat(degrees, "°", minutes, "′", seconds, "″", sign);
  }
};

REGISTER_WIKI_MACRO("coord", CoordinateTemplate);

// Template macro for countries.
class FlagTemplate : public WikiMacro {
 public:
  void Init(const Frame &config) override {
    Store *store = config.store();
    Frame country_codes(store, "/w/countries");
    if (country_codes.valid()) {
      for (const Slot &s : country_codes) {
        if (!store->IsString(s.name)) continue;
        string code = String(store, s.name).value();
        countries_[code] = s	.value;
      }
    }
  }

  void Generate(const WikiTemplate &templ, WikiAnnotator *annotator) override {
    // Look up country.
    string country = templ.GetValue(1);
    Handle item = Handle::nil();
    auto f = countries_.find(country);
    if (f != countries_.end()) {
      item = f->second;
    } else {
      Text qid = annotator->resolver()->ResolveLink(country);
      if (!qid.empty()) {
        item = annotator->store()->Lookup(qid);
      }
    }

    // Output country name.
    int begin = annotator->position();
    auto *namearg = templ.GetArgument("name");
    if (namearg != nullptr) {
      templ.Extract(namearg);
    } else {
      annotator->Content(country);
    }
    int end = annotator->position();

    // Output annotation.
    if (!item.IsNil()) {
      annotator->AddMention(begin, end, item);
    }
  }

 private:
  std::unordered_map<string, Handle> countries_;
};

REGISTER_WIKI_MACRO("flag", FlagTemplate);

}  // namespace nlp
}  // namespace sling

