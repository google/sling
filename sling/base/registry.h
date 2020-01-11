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

// Registry for component registration. These classes can be used for creating
// registries of components conforming to the same interface. This is useful for
// making a component-based architecture where the specific implementation
// classes can be selected at runtime. There is support for both class-based and
// instance based registries.
//
// Example:
//  function.h:
//
//   class Function : public Component<Function> {
//    public:
//     virtual double Evaluate(double x) = 0;
//   };
//
//   #define REGISTER_FUNCTION(type, component)
//     REGISTER_COMPONENT_TYPE(Function, type, component);
//
//  function.cc:
//
//   REGISTER_COMPONENT_REGISTRY("function", Function);
//
//   class Cos : public Function {
//    public:
//     double Evaluate(double x) { return cos(x); }
//   };
//
//   class Exp : public Function {
//    public:
//     double Evaluate(double x) { return exp(x); }
//   };
//
//   REGISTER_FUNCTION("cos", Cos);
//   REGISTER_FUNCTION("exp", Exp);
//
//   Function *f = Function::Create("cos");
//   double result = f->Evaluate(arg);
//   delete f;

#ifndef SLING_BASE_REGISTRY_H_
#define SLING_BASE_REGISTRY_H_

#include <string.h>
#include <functional>
#include <string>
#include <vector>

#include "sling/base/logging.h"
#include "sling/base/types.h"

namespace sling {

// Component metadata with information about name, class, and code location.
class ComponentMetadata {
 public:
  ComponentMetadata(const char *name, const char *class_name, const char *file,
                    int line)
      : name_(name),
        class_name_(class_name),
        file_(file),
        line_(line),
        link_(nullptr) {}

  // Returns component name.
  const char *name() const { return name_; }

  // Returns class name.
  const char *class_name() const { return class_name_; }

  // Returns file name.
  const char *file() const { return file_; }

  // Returns line.
  int line() const { return line_; }

  // Metadata objects can be linked in a list.
  ComponentMetadata *link() const { return link_; }
  void set_link(ComponentMetadata *link) { link_ = link; }

 private:
  // Component name.
  const char *name_;

  // Name of class for component.
  const char *class_name_;

  // Code file and location where the component was registered.
  const char *file_;
  int line_;

  // Link to next metadata object in list.
  ComponentMetadata *link_;
};

// The master registry contains all registered component registries. A registry
// is not registered in the master registry until the first component of that
// type is registered.
class RegistryMetadata : public ComponentMetadata {
 public:
  RegistryMetadata(const char *name, const char *class_name, const char *file,
                   int line, ComponentMetadata **components)
      : ComponentMetadata(name, class_name, file, line),
        components_(components) {}

  // Returns a list of components in the registry.
  void GetComponents(std::vector<const ComponentMetadata *> *components) const;

  // Returns meta data for component in registry. Returns null if not found.
  const ComponentMetadata *GetComponent(const string &name) const;

  // Returns the next registry in the registry list.
  RegistryMetadata *next() const {
    return static_cast<RegistryMetadata *>(link());
  }

  // Registers a component registry in the master registry.
  static void Register(RegistryMetadata *registry);

  // Returns list of all registries.
  static void GetRegistries(std::vector<const RegistryMetadata *> *registries);

  // Returns registry for component type or null if it is not found.
  static const RegistryMetadata *GetRegistry(const string &name);

 private:
  // Location of list of components in registry.
  ComponentMetadata **components_;

  // List of all component registries.
  static RegistryMetadata *global_registry_list;
};

// Registry for components. An object can be registered with a type name in the
// registry. The named instances in the registry can be returned using the
// Lookup() method. The components in the registry are put into a linked list
// of components. It is important that the component registry can be statically
// initialized in order not to depend on initialization order.
template <class T> struct ComponentRegistry {
  typedef ComponentRegistry<T> Self;

  // Component registration class.
  class Registrar : public ComponentMetadata {
   public:
    // Registers new component by linking itself into the component list of
    // the registry.
    Registrar(Self *registry, const char *type, const char *class_name,
              const char *file, int line, T *object)
        : ComponentMetadata(type, class_name, file, line), object_(object) {
      // Register registry in master registry if this is the first registered
      // component of this type.
      if (registry->components == nullptr) {
        RegistryMetadata::Register(new RegistryMetadata(
            registry->name, registry->class_name, registry->file,
            registry->line,
            reinterpret_cast<ComponentMetadata **>(&registry->components)));
      }

      // Register component in registry.
      set_link(registry->components);
      registry->components = this;
    }

    // Returns component type.
    const char *type() const { return name(); }

    // Returns component object.
    T *object() const { return object_; }

    // Returns the next component in the component list.
    Registrar *next() const { return static_cast<Registrar *>(link()); }

   private:
    // Component object.
    T *object_;
  };

  // Finds registrar for named component in registry.
  const Registrar *GetComponent(const char *type) const {
    Registrar *r = components;
    while (r != nullptr && strcmp(type, r->type()) != 0) r = r->next();
    if (r == nullptr) {
      LOG(FATAL) << "Unknown " << name << " component: " << type;
    }
    return r;
  }

  // Finds a named component in the registry.
  T *Lookup(const char *type) const { return GetComponent(type)->object(); }
  T *Lookup(const string &type) const { return Lookup(type.c_str()); }

  // Textual description of the kind of components in the registry.
  const char *name;

  // Base class name of component type.
  const char *class_name;

  // File and line where the registry is defined.
  const char *file;
  int line;

  // Linked list of registered components.
  Registrar *components;
};

// Base class for registerable components.
template <class T> class Component {
 public:
  // Factory function type.
  typedef std::function<T *()> Factory;

  // Registry type.
  typedef ComponentRegistry<Factory> Registry;

  // Creates a new component instance.
  static T *Create(const string &type) {
    return (*registry()->Lookup(type))();
  }

  // Returns registry for class.
  static Registry *registry() { return &registry_; }

 private:
  // Registry for class.
  static Registry registry_;
};

// Base class for registerable singletons.
template <class T> class Singleton {
 public:
  // Registry type.
  typedef ComponentRegistry<T> Registry;

  // Returns registry for class.
  static Registry *registry() { return &registry_; }

 private:
  // Registry for class.
  static Registry registry_;
};

#define REGISTER_COMPONENT_TYPE(base, type, component) \
  static base::Factory __##component##_factory = [] { return new component; }; \
  __attribute__((init_priority(800))) \
  static base::Registry::Registrar __##component##__##registrar( \
      base::registry(), type, #component, __FILE__, __LINE__, \
      &__##component##_factory)

#define REGISTER_COMPONENT_REGISTRY(type, classname) \
  template <> __attribute__((init_priority(900))) \
  classname::Registry sling::Component<classname>::registry_ = { \
      type, #classname, __FILE__, __LINE__, nullptr}

#define REGISTER_SINGLETON_TYPE(base, type, component) \
  __attribute__((init_priority(800))) \
  static base::Registry::Registrar __##component##__##registrar( \
      base::registry(), type, #component, __FILE__, __LINE__, new component)

#define REGISTER_SINGLETON_REGISTRY(type, classname) \
  template <> __attribute__((init_priority(900))) \
  classname::Registry sling::Singleton<classname>::registry_ = { \
      type, #classname, __FILE__, __LINE__, nullptr}

}  // namespace sling

#endif  // SLING_BASE_REGISTRY_H_
