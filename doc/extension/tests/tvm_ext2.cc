// #include <tvm/runtime/device_api.h>
// #include <tvm/runtime/module.h>
// #include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
// #include <tvm/tir/op.h>

// using namespace tvm;
using namespace tvm::runtime;
// namespace tvm_ext {
// TVM_REGISTER_GLOBAL("tvm_ext.sym_add").set_body([](TVMArgs args, TVMRetValue* rv) {
//   tvm::tir::Var a = args[0];
//   tvm::tir::Var b = args[1];
//   *rv = a + b;
// });

// TVM_REGISTER_GLOBAL("tvm_ext.bind_add").set_body([](TVMArgs args_, TVMRetValue* rv_) {
//   PackedFunc pf = args_[0];
//   int b = args_[1];
//   *rv_ = PackedFunc([pf, b](TVMArgs args, TVMRetValue* rv) { *rv = pf(b, args[0]); });
// });

// TVM_REGISTER_GLOBAL("device_api.ext_dev").set_body([](TVMArgs args, TVMRetValue* rv) {
//   *rv = (*tvm::runtime::Registry::Get("device_api.cpu"))();
// });

// } // namespace tvm_ext

// 暴露给运行时的外部函数
extern "C" float TVMTestAddOne(float y) { return y + 1; }

// 这个回调方法允许扩展，使 TVM 能够提取。
// 当想要使用仅包含头文件的最小版本的 TVM 运行时，这种方法会很有帮助。
extern "C" int TVMExtDeclare(TVMFunctionHandle pregister) {
  const PackedFunc& fregister = GetRef<PackedFunc>(static_cast<PackedFuncObj*>(pregister));
  // 等价于 const PackedFunc& fregister = *static_cast<PackedFunc*>(pregister);
  auto mul = [](TVMArgs args, TVMRetValue* rv) {
    int x = args[0];
    int y = args[1];
    *rv = x * y;
  };
  fregister("mul", PackedFunc(mul));
  return 0;
}