#include <tvm/runtime/registry.h>

// 暴露给运行时的外部函数
extern "C" float TVMTestAddOne(float y) { return y + 1; }
