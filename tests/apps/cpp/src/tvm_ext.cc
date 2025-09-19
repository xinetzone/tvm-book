#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

using namespace tvm::runtime;

namespace tvm_ext {
TVM_REGISTER_GLOBAL("tvm_ext.testing.bind_add").set_body([](TVMArgs args_, TVMRetValue* rv_) {
  PackedFunc pf = args_[0];
  int b = args_[1];
  *rv_ = PackedFunc([pf, b](TVMArgs args, TVMRetValue* rv) { *rv = pf(b, args[0]); });
});

} // namespace tvm_ext
