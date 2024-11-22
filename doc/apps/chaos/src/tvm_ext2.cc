#include <iostream>
#include <vector>
#include <string.h>
#include <sstream>
#include <cassert> // 断言
#include <iostream>
// #include <random>
// #include <algorithm>
// #include <cmath>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/relay/op_attr_types.h>

namespace tvm {
namespace runtime {
class BaseObj :public Object {
public:
    // 对象字段
    int field0;

    // 对象属性
    static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
    static constexpr const char* _type_key = "test.BaseObj";
    // 告诉 TVM 编译器，BaseObj 类是 Object 类的子类，并且需要在编译时进行一些特殊的处理。
    TVM_DECLARE_BASE_OBJECT_INFO(BaseObj, Object);
    void VisitAttrs(AttrVisitor* v) {
      v->Visit("field0", &field0);
  }
};

class LeafObj :public BaseObj {
public:
    // 字段
    int child_field0;
    // 对象属性
    static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
    static constexpr const char* _type_key = "test.LeafObj";
    TVM_DECLARE_BASE_OBJECT_INFO(LeafObj, Object);
};
TVM_REGISTER_NODE_TYPE(BaseObj);
}
}