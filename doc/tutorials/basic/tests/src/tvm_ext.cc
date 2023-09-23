#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
using namespace tvm::runtime;


// 参考：https://daobook.github.io/tvm/docs/arch/runtime.html
void MyAdd(TVMArgs args, TVMRetValue* rv) {
  // 自动将参数转换为所需的类型。
  int a = args[0];
  int b = args[1];
  // 自动分配返回值 rv
  *rv = a + b;
}

// 注册全局 packed function
TVM_REGISTER_GLOBAL("myadd").set_body(MyAdd);

TVM_REGISTER_GLOBAL("callhello")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  PackedFunc f = args[0];
  f("hello world");
});