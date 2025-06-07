#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/registry.h>

using namespace tvm::runtime;
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