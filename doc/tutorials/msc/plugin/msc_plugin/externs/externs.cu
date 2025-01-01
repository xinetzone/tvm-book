#include "externs.h"

#define CU1DBLOCK 256
#define KERNEL_LOOP(i, n)   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

namespace tvm {
namespace contrib {
namespace msc {
namespace plugin {

inline int n_blocks(int size, int block_size) {
  return size / block_size + (size % block_size == 0 ? 0 : 1);
}

template <typename T>
__global__ static void _my_relu(const T* src, T* dst, T max_val, int n) {
  KERNEL_LOOP(i, n) {
    if (src[i] >= max_val) {
      dst[i] = max_val;
    } else if (src[i] <= 0) {
      dst[i] = 0;
    } else {
      dst[i] = src[i];
    }
  }
}

template <typename T>
void my_relu_cuda_kernel(const DataTensor<T>& input, DataTensor<T>& output, T max_val,
                         const cudaStream_t& stream) {
  const T* input_data = input.const_data();
  T* output_data = output.data();
  dim3 Bl(CU1DBLOCK);
  dim3 Gr(n_blocks(output.size(), CU1DBLOCK));
  _my_relu<<<Gr, Bl, 0, stream>>>(input_data, output_data, max_val, output.size());
}

template void my_relu_cuda_kernel<float>(const DataTensor<float>& input, DataTensor<float>& output,
                                         float max_val, const cudaStream_t& stream);

}  // namespace plugin
}  // namespace msc
}  // namespace contrib
}  // namespace tvm
