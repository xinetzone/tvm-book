#include <tvm/runtime/device_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
// #include <tvm/tir/op.h>

using namespace tvm::runtime;

namespace tvm_ext {
TVM_REGISTER_GLOBAL("device_api.ext_dev").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = (*tvm::runtime::Registry::Get("device_api.cpu"))();
});

} // namespace tvm_ext