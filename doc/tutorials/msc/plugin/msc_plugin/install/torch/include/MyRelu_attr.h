#ifndef TVM_CONTRIB_MSC_MYRELU_ATTR_H_
#define TVM_CONTRIB_MSC_MYRELU_ATTR_H_

#include "plugin_utils.h"

namespace tvm {
namespace contrib {
namespace msc {
namespace plugin {

struct MyReluMetaAttr {
  // define attributes
  float max_val;
  
  // print method
  friend std::ostream& operator<<(std::ostream& out, const MyReluMetaAttr& attrs) {
    out << "[MyReluMetaAttr] : ";
    out << "| max_val(float)=" << attrs.max_val;
    return out;
  }

};  // struct MyReluMetaAttr

// serialize method
std::vector<std::string> MyReluMetaAttr_serialize(const MyReluMetaAttr& meta_attr);

// deserialize method
void MyReluMetaAttr_deserialize(const std::vector<std::string>& attrs, MyReluMetaAttr& meta_attr);

}  // namespace plugin
}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_MYRELU_ATTR_H_