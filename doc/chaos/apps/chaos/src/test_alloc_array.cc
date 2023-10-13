/*!
 * \brief 加载并运行 TVM module.s 的示例代码
 * \file test_alloc_array.cc
 */
#include <dlpack/dlpack.h>
#include <tvm/runtime/registry.h>

void pack_data() {
  // Allocate DLPack 数据结构。分配 nd-array 的内存，包括指定规格的形状空间。
  //
  // 注意，在这个例子中，使用 TVM 运行时 API 来分配 DLTensor。
  // TVM 接受 DLPack 兼容的 DLTensor，
  // 所以只要传递正确的 DLTensor 数组指针，函数就可以被调用。
  //
  // 更多信息请参考 dlpack。
  // 需要注意的一点是，DLPack 包含数据指针的 alignment requirement，而 TVM 利用了这一点。
  // 如果你计划使用你的自定义数据容器，请确保你传入的 DLTensor 符合 alignment requirement。
  //
  DLTensor* x;
  int ndim = 1; // 数组的维数
  int dtype_code = kDLFloat; // dtype 的类型代码
  int dtype_bits = 32; // dtype 的位数
  int dtype_lanes = 1; // dtype 的 lanes 数
  int device_type = kDLCPU; // 设备类型
  int device_id = 0; // 设备 ID
  // The shape of the array, the data content will be copied to out
  const tvm_index_t shape[1] = {10}; 
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
  // 数据加载
  for (int i = 0; i < shape[0]; ++i) {
    static_cast<float*>(x->data)[i] = i;
  }
  LOG(INFO) << "完成验证 DLTensor！";
  TVMArrayFree(x);
}

int main(void) {
  pack_data();
  return 0;
}
