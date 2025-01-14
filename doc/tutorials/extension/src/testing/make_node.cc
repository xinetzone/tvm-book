#include <string.h>
#include <tvm/runtime/object.h>
#include <tvm/node/reflection.h>

namespace tvm {
namespace runtime {
class TestNode :public Object {
public:
    // 对象字段
    std::string name;
    // 对象属性
    static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
    static constexpr const char* _type_key = "app.TestNode";
    TVM_DECLARE_BASE_OBJECT_INFO(TestNode, Object);
    void VisitAttrs(AttrVisitor* v) {
        v->Visit("name", &name);
    }
};
TVM_REGISTER_NODE_TYPE(TestNode); // 注册节点类型
}
}