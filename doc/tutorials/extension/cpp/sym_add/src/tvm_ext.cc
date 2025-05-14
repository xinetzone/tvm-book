#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/op.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/function.h>

namespace tvm_ext {
using namespace tvm::ffi;
TVM_REGISTER_GLOBAL("tvm_ext.sym_add")
    .set_body_packed([](const AnyView* args, int32_t num_args, Any* rv) {
      // *rv = args[0] + args[1];
      int32_t a = args[0].cast<int32_t>();
      *rv = a + 1;
    });
} // namespace tvm_ext
