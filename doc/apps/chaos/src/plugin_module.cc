/*!
 * \brief 可以由 TVM 运行时编译和加载的示例代码。
 * \file plugin_module.cc
 */
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

namespace tvm_dso_plugin {

using namespace tvm::runtime;

class MyModuleNode : public ModuleNode {
 public:
  explicit MyModuleNode(int value) : value_(value) {}

  virtual const char* type_key() const final { return "MyModule"; }

  virtual PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "add") {
      return TypedPackedFunc<int(int)>([sptr_to_self, this](int value) { return value_ + value; });
    } else if (name == "mul") {
      return TypedPackedFunc<int(int)>([sptr_to_self, this](int value) { return value_ * value; });
    } else {
      LOG(FATAL) << "unknown function " << name;
    }
  }

 private:
  int value_;
};

void CreateMyModule_(TVMArgs args, TVMRetValue* rv) {
  int value = args[0];
  *rv = Module(make_object<MyModuleNode>(value));
}

int SubOne_(int x) { return x - 1; }

// USE TVM_DLL_EXPORT_TYPED_PACKED_FUNC to export a
// typed function as packed function.
TVM_DLL_EXPORT_TYPED_FUNC(SubOne, SubOne_);

// TVM_DLL_EXPORT_TYPED_PACKED_FUNC also works for lambda.
TVM_DLL_EXPORT_TYPED_FUNC(AddOne, [](int x) -> int { return x + 1; });

// Use TVM_EXPORT_PACKED_FUNC to export a function with
TVM_DLL_EXPORT_PACKED_FUNC(CreateMyModule, tvm_dso_plugin::CreateMyModule_);
}  // namespace tvm_dso_plugin
