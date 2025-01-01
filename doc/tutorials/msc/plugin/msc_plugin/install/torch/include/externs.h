#ifndef EXTERNS_H_
#define EXTERNS_H_

#include "plugin_base.h"

#ifdef PLUGIN_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace tvm {
namespace contrib {
namespace msc {
namespace plugin {

template <typename TAttr>
std::vector<MetaTensor> my_relu_infer(const std::vector<MetaTensor>& inputs, const TAttr& attrs,
                                      bool is_runtime) {
  std::vector<MetaTensor> outputs;
  outputs.push_back(MetaTensor(inputs[0].shape(), inputs[0].data_type(), inputs[0].layout()));
  return outputs;
}

template <typename T>
void my_relu_cpu_kernel(const DataTensor<T>& input, DataTensor<T>& output, T max_val);

template <typename T, typename TAttr>
void my_relu_cpu_compute(const DataTensor<T>& input, DataTensor<T>& output, const TAttr& attrs) {
  my_relu_cpu_kernel(input, output, T(attrs.max_val));
}

#ifdef PLUGIN_ENABLE_CUDA
template <typename T>
void my_relu_cuda_kernel(const DataTensor<T>& input, DataTensor<T>& output, T max_val,
                         const cudaStream_t& stream);

template <typename T, typename TAttr>
void my_relu_cuda_compute(const DataTensor<T>& input, DataTensor<T>& output, const TAttr& attrs,
                          const cudaStream_t& stream) {
  my_relu_cuda_kernel(input, output, T(attrs.max_val), stream);
}
#endif

}  // namespace plugin
}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // EXTERNS_H_
