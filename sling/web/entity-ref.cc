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

#include "sling/web/entity-ref.h"

#include "sling/base/macros.h"
#include "sling/base/types.h"

namespace sling {

struct EntityRef {
  int code;
  const char *name;
};

static EntityRef enttab[] = {
  { 198, "AElig"},
  { 193, "Aacute"},
  { 194, "Acirc"},
  { 192, "Agrave"},
  { 913, "Alpha"},
  { 197, "Aring"},
  { 195, "Atilde"},
  { 196, "Auml"},
  { 914, "Beta"},
  { 199, "Ccedil"},
  { 935, "Chi"},
  {8225, "Dagger"},
  { 916, "Delta"},
  { 208, "ETH"},
  { 201, "Eacute"},
  { 202, "Ecirc"},
  { 200, "Egrave"},
  { 917, "Epsilon"},
  { 919, "Eta"},
  { 203, "Euml"},
  { 915, "Gamma"},
  { 205, "Iacute"},
  { 206, "Icirc"},
  { 204, "Igrave"},
  { 921, "Iota"},
  { 207, "Iuml"},
  { 922, "Kappa"},
  { 923, "Lambda"},
  { 924, "Mu"},
  { 209, "Ntilde"},
  { 925, "Nu"},
  { 338, "OElig"},
  { 211, "Oacute"},
  { 212, "Ocirc"},
  { 210, "Ograve"},
  { 937, "Omega"},
  { 927, "Omicron"},
  { 216, "Oslash"},
  { 213, "Otilde"},
  { 214, "Ouml"},
  { 934, "Phi"},
  { 928, "Pi"},
  {8243, "Prime"},
  { 936, "Psi"},
  { 929, "Rho"},
  { 352, "Scaron"},
  { 931, "Sigma"},
  { 222, "THORN"},
  { 932, "Tau"},
  { 920, "Theta"},
  { 218, "Uacute"},
  { 219, "Ucirc"},
  { 217, "Ugrave"},
  { 933, "Upsilon"},
  { 220, "Uuml"},
  { 926, "Xi"},
  { 221, "Yacute"},
  { 376, "Yuml"},
  { 918, "Zeta"},
  { 225, "aacute"},
  { 226, "acirc"},
  { 180, "acute"},
  { 230, "aelig"},
  { 224, "agrave"},
  {8501, "alefsym"},
  { 945, "alpha"},
  {  38, "amp"},
  {8743, "and"},
  {8736, "ang"},
  {  39, "apos"},
  { 229, "aring"},
  {8776, "asymp"},
  { 227, "atilde"},
  { 228, "auml"},
  {8222, "bdquo"},
  { 946, "beta"},
  { 166, "brvbar"},
  {8226, "bull"},
  {8745, "cap"},
  { 231, "ccedil"},
  { 184, "cedil"},
  { 162, "cent"},
  { 967, "chi"},
  { 710, "circ"},
  {9827, "clubs"},
  {8773, "cong"},
  { 169, "copy"},
  {8629, "crarr"},
  {8746, "cup"},
  { 164, "curren"},
  {8659, "dArr"},
  {8224, "dagger"},
  {8595, "darr"},
  { 176, "deg"},
  { 948, "delta"},
  {9830, "diams"},
  { 247, "divide"},
  { 233, "eacute"},
  { 234, "ecirc"},
  { 232, "egrave"},
  {8709, "empty"},
  {8195, "emsp"},
  {8194, "ensp"},
  { 949, "epsilon"},
  {8801, "equiv"},
  { 951, "eta"},
  { 240, "eth"},
  { 235, "euml"},
  {8364, "euro"},
  {8707, "exist"},
  { 402, "fnof"},
  {8704, "forall"},
  { 189, "frac12"},
  { 188, "frac14"},
  { 190, "frac34"},
  {8260, "frasl"},
  { 947, "gamma"},
  {8805, "ge"},
  {  62, "gt"},
  {8660, "hArr"},
  {8596, "harr"},
  {9829, "hearts"},
  {8230, "hellip"},
  { 237, "iacute"},
  { 238, "icirc"},
  { 161, "iexcl"},
  { 236, "igrave"},
  {8465, "image"},
  {8734, "infin"},
  {8747, "int"},
  { 953, "iota"},
  { 191, "iquest"},
  {8712, "isin"},
  { 239, "iuml"},
  { 954, "kappa"},
  {8656, "lArr"},
  { 955, "lambda"},
  {9001, "lang"},
  { 171, "laquo"},
  {8592, "larr"},
  {8968, "lceil"},
  {8220, "ldquo"},
  {8804, "le"},
  {8970, "lfloor"},
  {8727, "lowast"},
  {9674, "loz"},
  {8206, "lrm"},
  {8249, "lsaquo"},
  {8216, "lsquo"},
  {  60, "lt"},
  { 175, "macr"},
  {8212, "mdash"},
  { 181, "micro"},
  { 183, "middot"},
  {8722, "minus"},
  { 956, "mu"},
  {8711, "nabla"},
  { 160, "nbsp"},
  {8211, "ndash"},
  {8800, "ne"},
  {8715, "ni"},
  { 172, "not"},
  {8713, "notin"},
  {8836, "nsub"},
  { 241, "ntilde"},
  { 957, "nu"},
  { 243, "oacute"},
  { 244, "ocirc"},
  { 339, "oelig"},
  { 242, "ograve"},
  {8254, "oline"},
  { 969, "omega"},
  { 959, "omicron"},
  {8853, "oplus"},
  {8744, "or"},
  { 170, "ordf"},
  { 186, "ordm"},
  { 248, "oslash"},
  { 245, "otilde"},
  {8855, "otimes"},
  { 246, "ouml"},
  { 182, "para"},
  {8706, "part"},
  {8240, "permil"},
  {8869, "perp"},
  { 966, "phi"},
  { 960, "pi"},
  { 982, "piv"},
  { 177, "plusmn"},
  { 163, "pound"},
  {8242, "prime"},
  {8719, "prod"},
  {8733, "prop"},
  { 968, "psi"},
  {  34, "quot"},
  {8658, "rArr"},
  {8730, "radic"},
  {9002, "rang"},
  { 187, "raquo"},
  {8594, "rarr"},
  {8969, "rceil"},
  {8221, "rdquo"},
  {8476, "real"},
  { 174, "reg"},
  {8971, "rfloor"},
  { 961, "rho"},
  {8207, "rlm"},
  {8250, "rsaquo"},
  {8217, "rsquo"},
  {8218, "sbquo"},
  { 353, "scaron"},
  {8901, "sdot"},
  { 167, "sect"},
  { 173, "shy"},
  { 963, "sigma"},
  { 962, "sigmaf"},
  {8764, "sim"},
  {9824, "spades"},
  {8834, "sub"},
  {8838, "sube"},
  {8721, "sum"},
  {8835, "sup"},
  { 185, "sup1"},
  { 178, "sup2"},
  { 179, "sup3"},
  {8839, "supe"},
  { 223, "szlig"},
  { 964, "tau"},
  {8756, "there4"},
  { 952, "theta"},
  { 977, "thetasym"},
  {8201, "thinsp"},
  { 254, "thorn"},
  { 732, "tilde"},
  { 215, "times"},
  {8482, "trade"},
  {8657, "uArr"},
  { 250, "uacute"},
  {8593, "uarr"},
  { 251, "ucirc"},
  { 249, "ugrave"},
  { 168, "uml"},
  { 978, "upsih"},
  { 965, "upsilon"},
  { 252, "uuml"},
  {8472, "weierp"},
  { 958, "xi"},
  { 253, "yacute"},
  { 165, "yen"},
  { 255, "yuml"},
  { 950, "zeta"},
  {8205, "zwj"},
  {8204, "zwnj"}
};

int ParseEntityRef(const char *str, int len, int *consumed) {
  const char *p = str;
  const char *end = str + len;
  int code = 0;

  if (*p++ != '&') return -1;
  if (p >= end) return -1;

  if (*p == '#') {
    if (++p == end) return -1;

    if (*p == 'x') {
      // &#x10FF; (hexadecimal)
      if (++p == end) return -1;
      while (p < end) {
        if (*p >= '0' && *p <= '9') {
          code = code * 16 + (*p++ - '0');
        } else if (*p >= 'A' && *p <= 'F') {
          code = code * 16 + (*p++ - 'A' + 10);
        } else if (*p >= 'a' && *p <= 'f') {
          code = code * 16 + (*p++ - 'a' + 10);
        } else if (*p == ';') {
          p++;
          break;
        } else {
          return -1;
        }
      }
    } else {
      // &#x123; (decimal)
      while (p < end) {
        if (*p >= '0' && *p <= '9') {
          code = code * 10 + (*p++ - '0');
        } else if (*p == ';') {
          p++;
          break;
        } else {
          return -1;
        }
      }
    }

    *consumed = p - str;
    return code;
  } else {
    // &name; (named entity)
    const char *name = p;
    while (p < end && *p != ';') p++;
    if (p == end) return -1;
    int len = p++ - name;

    // Look up entity reference name.
    int lo = 0;
    int hi = ARRAYSIZE(enttab) - 1;
    while (lo <= hi) {
      int mid = (lo + hi) / 2;
      int cmp = strncmp(name, enttab[mid].name, len);
      if (cmp == 0 && strlen(enttab[mid].name) == len) {
        *consumed = p - str;
        return enttab[mid].code;
      }
      if (cmp < 0) {
        hi = mid - 1;
      } else {
        lo = mid + 1;
      }
    }
    return -1;
  }
}

int ParseEntityRef(const string &str) {
  int consumed;
  int code = ParseEntityRef(str.data(), str.size(), &consumed);
  if (consumed != str.size()) return -1;
  return code;
}

}  // namespace sling

