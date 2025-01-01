#ifndef TVM_CONTRIB_MSC_MYRELU_OP_H_
#define TVM_CONTRIB_MSC_MYRELU_OP_H_

#include "MyRelu_attr.h"
#include "externs.h"

namespace tvm {
namespace contrib {
namespace msc {
namespace plugin {

struct MyRelu : torch::CustomClassHolder {
  MyRelu(const std::vector<std::string>& attrs);

  // serialize method
  const std::vector<std::string> serialize();

  // main compute
  std::vector<torch::Tensor> compute(const std::vector<torch::Tensor>& input_tensors);

  // members
  MyReluMetaAttr meta_attr_;
  std::vector<MetaLayout> layouts_;
  std::string name_;
};  // struct MyRelu : torch::CustomClassHolder

// Entry method for plugin MyRelu
std::vector<torch::Tensor> my_relu_entry(const c10::intrusive_ptr<MyRelu>& instance, const torch::Tensor& input, const double& max_val, const std::string& name);

}  // namespace plugin
}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_MYRELU_OP_H_