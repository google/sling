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

#ifndef SLING_UTIL_ASSET_H_
#define SLING_UTIL_ASSET_H_

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>

namespace sling {

// Shared asset.
class Asset {
 public:
  virtual ~Asset() = default;
};

// Return unique identifier for type.
typedef size_t TypeID;
template<typename T> inline TypeID TypeId() {
 static char signature;
 return reinterpret_cast<TypeID>(&signature);
}

// Asset manager that can hold one shared instance per type id and asset name
// combination.
class AssetManager {
 public:
  ~AssetManager() {
    for (auto &it : assets_) delete it.second;
  }

  // Delete all assets.
  void DisposeAssets() {
    for (auto &it : assets_) delete it.second;
    assets_.clear();
  }

  // Return asset for type and name, initializing a new instance the first
  // time the type and name pair is acquired.
  template<class T> const T *Acquire(
      const std::string &name,
      std::function<T *()> init) {
    Key key(TypeId<T>(), name);
    Asset *&a = assets_[key];
    if (a == nullptr) a = init();
    return reinterpret_cast<const T *>(a);
  }

 private:
  // An asset key consists of a type id and an asset name.
  typedef std::pair<TypeID, std::string> Key;

  struct KeyHash {
    size_t operator()(const Key &key) const {
      size_t h1 = std::hash<sling::TypeID>()(key.first);
      size_t h2 = std::hash<string>()(key.second);
      return  h1 ^ h2;
    }
  };

  // Mapping from type id and asset name to asset.
  std::unordered_map<Key, Asset *, KeyHash> assets_;
};

}  // namespace sling

#endif  // SLING_UTIL_ASSET_H_

