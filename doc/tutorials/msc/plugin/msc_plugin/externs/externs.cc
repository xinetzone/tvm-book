#include "externs.h"

namespace tvm {
namespace contrib {
namespace msc {
namespace plugin {

template <typename T>
void my_relu_cpu_kernel(const DataTensor<T>& input, DataTensor<T>& output, T max_val) {
  const T* input_data = input.const_data();
  T* output_data = output.data();
  for (size_t i = 0; i < output.size(); i++) {
    if (input_data[i] >= max_val) {
      output_data[i] = max_val;
    } else if (input_data[i] <= 0) {
      output_data[i] = 0;
    } else {
      output_data[i] = input_data[i];
    }
  }
}

template void my_relu_cpu_kernel<float>(const DataTensor<float>& input, DataTensor<float>& output,
                                        float max_val);

}  // namespace plugin
}  // namespace msc
}  // namespace contrib
}  // namespace tvm
