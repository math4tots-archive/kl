#ifndef orange_hpp
#define orange_hpp
#include <map>
#include <memory>
#include <stddef.h>
#include <string>
#include <functional>
#include <vector>

namespace orange {

using Id = size_t;
class Value;
using Pointer = std::shared_ptr<Value>;  // TODO: Handle stackoverflow

inline Pointer eval(Pointer scope, Pointer code);

inline std::string Error(const std::string &message) {
  return message;
}

class State {
public:
  std::vector<std::string> id_to_name;
  std::map<std::string, Id> name_to_id;

  Id opcode_literal = get_id("literal");
  Id opcode_getvar = get_id("getvar");
  Id opcode_setvar = get_id("setvar");
  Id opcode_block = get_id("block");

  Id id_this = get_id("this");

  Pointer null_proto;
  Pointer number_proto;
  Pointer symbol_proto;
  Pointer list_proto;
  Pointer function_proto;

  Pointer null_value;

  Id get_id(const std::string &str) {
    // Search existing ids
    auto i = name_to_id.find(str);
    if (i != name_to_id.end()) {
      return i->second;
    }
    // Since existing could not be found, create new id
    Id s = id_to_name.size();
    id_to_name.push_back(str);
    name_to_id[str] = s;
    return s;
  }

  const std::string &get_name(Id s) {
    return id_to_name.at(s);
  }
};

inline State state;

inline Id get_id(const std::string &str) {
  return state.get_id(str);
}

inline const std::string &get_name(Id s) {
  return state.get_name(s);
}

class Value : public std::enable_shared_from_this<Value> {
public:
  virtual ~Value() {}
  virtual Pointer *findattr(Id)=0;

  virtual void setattr(Id, const Pointer&) {
    throw Error("setattr not allowed for this type");
  }

  bool hasattr(Id id) {
    return findattr(id);
  }

  const Pointer &getattr(Id id) {
    Pointer *p = findattr(id);
    if (!p) {
      throw Error("Could not getattr");
    }
    return *p;
  }

  virtual Pointer call(const Pointer&, const std::vector<Pointer>&) {
    throw Error("Not a function");
  }

  virtual Pointer bind(Pointer) {
    return shared_from_this();
  }

  virtual bool is_null() const {
    return false;
  }

  virtual double number_value() const {
    throw Error("Expected Number");
  }

  virtual Id symbol_value() const {
    throw Error("Expected Symbol");
  }

  virtual std::vector<Pointer> &list_value() {
    throw Error("Expected List");
  }
};

class Object final : public Value {
  const Pointer proto;
  std::map<Id, Pointer> table;
public:
  Object(Pointer p): proto(p) {}

  Pointer *findattr(Id id) override {
    auto i = table.find(id);
    if (i != table.end()) {
      return &i->second;
    } else if (proto) {
      return proto->findattr(id);
    }
    return NULL;
  }
};

class PrimitiveValue : public Value {
public:
  virtual Pointer getproto()=0;

  Pointer *findattr(Id id) override {
    return getproto()->findattr(id);
  }
};

class Null final : public PrimitiveValue {
public:
  static Pointer get() {
    if (!state.null_value) {
      state.null_value = std::make_shared<Null>();
    }
    return state.null_value;
  }

  Pointer getproto() override {
    if (!state.null_proto) {
      state.null_proto = std::make_shared<Object>(Pointer());
    }
    return state.null_proto;
  }

  bool is_null() const override {
    return true;
  }
};

class Number final : public PrimitiveValue {
  double number;

public:
  Pointer getproto() override {
    if (!state.number_proto) {
      state.number_proto = std::make_shared<Object>(Pointer());
    }
    return state.number_proto;
  }

  double number_value() const override {
    return number;
  }
};

class Symbol final : public PrimitiveValue {
  Id id;

public:
  Pointer getproto() override {
    if (!state.symbol_proto) {
      state.symbol_proto = std::make_shared<Object>(Pointer());
    }
    return state.symbol_proto;
  }

  Id symbol_value() const override {
    return id;
  }
};

class List final : public PrimitiveValue {
  std::vector<Pointer> items;

public:
  Pointer getproto() override {
    if (!state.list_proto) {
      state.list_proto = std::make_shared<Object>(Pointer());
    }
    return state.list_proto;
  }

  std::vector<Pointer> &list_value() override {
    return items;
  }
};

class Function : public PrimitiveValue {
public:
  Pointer getproto() override {
    if (!state.function_proto) {
      state.function_proto = std::make_shared<Object>(Pointer());
    }
    return state.function_proto;
  }

  virtual Pointer bind(Pointer owner) override;
};

class BoundFunction final : public Function {
  Pointer owner;
  Pointer function;
public:
  BoundFunction(Pointer owner, Pointer fn): owner(owner), function(fn) {}

  Pointer call(const Pointer&, const std::vector<Pointer> &args)
      override {
    return function->call(owner, args);
  }

  Pointer bind(Pointer) override {
    return shared_from_this();
  }
};

inline Pointer Function::bind(Pointer owner) {
  return std::make_shared<BoundFunction>(owner, shared_from_this());
}

class BuiltinFunction final : public Function {
  using Impl = std::function<Pointer(Pointer, const std::vector<Pointer>&)>;

  const std::string name;
  Impl impl;

public:
  BuiltinFunction(const std::string &n, const Impl &f): name(n), impl(f) {}

  Pointer call(const Pointer &owner, const std::vector<Pointer> &args)
      override {
    return impl(owner, args);
  }
};

class NormalFunction final : public Function {
  Pointer scope;
  const std::vector<Id> argnames;
  Pointer body;

public:
  NormalFunction(const Pointer &sc, const std::vector<Id> &ps, Pointer b):
      scope(sc), argnames(ps), body(b) {}

  Pointer call(const Pointer &owner, const std::vector<Pointer> &args) override {
    if (args.size() != argnames.size()) {
      throw Error("Mismatched argc");
    }
    Pointer fscope = std::make_shared<Object>(scope);
    fscope->setattr(state.id_this, owner);
    for (size_t i = 0; i < argnames.size(); i++) {
      fscope->setattr(argnames[i], args[i]);
    }
    return eval(fscope, body);
  }
};

inline void check_op_length(std::vector<Pointer> &list, size_t len) {
  if (list.size() != len) {
    throw Error("Invalid list length");
  }
}

inline Pointer eval(Pointer scope, Pointer code) {
  Value *code_obj = code.get();
  List *code_list = dynamic_cast<List*>(code_obj);
  if (code_list) {
    std::vector<Pointer> &list = code_list->list_value();
    Id op = list.at(0)->symbol_value();
    if (op == state.opcode_literal) {
      check_op_length(list, 2);
      return list.at(1);
    } else if (op == state.opcode_getvar) {
      check_op_length(list, 2);
      Id name = list.at(1)->symbol_value();
      return scope->getattr(name);
    } else if (op == state.opcode_setvar) {
      check_op_length(list, 3);
      Id name = list.at(1)->symbol_value();
      Pointer value = list.at(2);
      scope->setattr(name, value);
      return value;
    } else if (op == state.opcode_block) {
      size_t size = list.size();
      Pointer last = Null::get();
      for (size_t i = 1; i < size; i++) {
        last = eval(scope, list[i]);
      }
      return last;
    }
    throw Error("Unrecognized opcode " + get_name(op));
  }
  throw Error("Expected code object to be list for eval");
}

}  // namespace orange
#endif//orange_hpp
