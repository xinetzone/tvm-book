#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <dlpack/dlpack.h>
#include <cstdint>
#include <cstring>
#include <iostream>

#define GCC_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_)           \
  extern "C" void p_ID_(float* a, float* b, float* out) { \
    for (int64_t i = 0; i < p_DIM1_; ++i) {               \
      out[i] = a[i] p_OP_ b[i];                           \
    }                                                     \
  }

#define GCC_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_)  \
  extern "C" void p_ID_(float* a, float* b, float* out) { \
    for (int64_t i = 0; i < p_DIM1_; ++i) {               \
      for (int64_t j = 0; j < p_DIM2_; ++j) {             \
        int64_t k = i * p_DIM2_ + j;                      \
        out[k] = a[k] p_OP_ b[k];                         \
      }                                                   \
    }                                                     \
  }

// Note 1
GCC_BINARY_OP_2D(gcc_0_0, *, 10, 10);
GCC_BINARY_OP_2D(gcc_0_1, -, 10, 10);
GCC_BINARY_OP_2D(gcc_0_2, +, 10, 10);

// Note 2
extern "C" void gcc_0_(float* gcc_input0, float* gcc_input1,
                       float* gcc_input2, float* gcc_input3, float* out) {
  float* buf_0 = (float*)malloc(4 * 100);
  float* buf_1 = (float*)malloc(4 * 100);
  gcc_0_2(gcc_input0, gcc_input1, buf_0);
  gcc_0_1(buf_0, gcc_input2, buf_1);
  gcc_0_0(buf_1, gcc_input3, out);
  free(buf_0);
  free(buf_1);
}

// Note 3
extern "C" int gcc_0_wrapper(DLTensor* arg0, DLTensor* arg1, DLTensor* arg2,
                             DLTensor* arg3, DLTensor* out) {
  gcc_0_(static_cast<float*>(arg0->data), static_cast<float*>(arg1->data),
         static_cast<float*>(arg2->data), static_cast<float*>(arg3->data),
         static_cast<float*>(out->data));
  return 0;
}
TVM_DLL_EXPORT_TYPED_FUNC(gcc_0, gcc_0_wrapper);