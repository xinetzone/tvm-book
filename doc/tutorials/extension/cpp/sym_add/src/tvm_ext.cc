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
} // namespace tvm_ext
