#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

using namespace tvm::runtime;

namespace tvm_ext {
TVM_REGISTER_GLOBAL("callhello")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  PackedFunc f = args[0];
  f("hello world");
});

} // namespace tvm_ext
