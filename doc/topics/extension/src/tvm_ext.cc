#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/op.h>

using namespace tvm::runtime;

namespace tvm_ext {
TVM_REGISTER_GLOBAL("tvm_ext.sym_add").set_body([](TVMArgs args, TVMRetValue* rv) {
  tvm::tir::Var a = args[0];
  tvm::tir::Var b = args[1];
  *rv = a + b;
});

// TVM_REGISTER_GLOBAL("tvm_ext.bind_add").set_body([](TVMArgs args_, TVMRetValue* rv_) {
//   PackedFunc pf = args_[0];
//   int b = args_[1];
//   *rv_ = PackedFunc([pf, b](TVMArgs args, TVMRetValue* rv) { *rv = pf(b, args[0]); });
// });

// TVM_REGISTER_GLOBAL("device_api.ext_dev").set_body([](TVMArgs args, TVMRetValue* rv) {
//   *rv = (*tvm::runtime::Registry::Get("device_api.cpu"))();
// });

} // namespace tvm_ext
