// #include <tvm/runtime/c_runtime_api.h>
// #include <dlpack/dlpack.h>
// #include <tvm/runtime/module.h>
// #include <tvm/runtime/registry.h>
// #include <tvm/runtime/packed_func.h>
// #include <tvm/runtime/logging.h>
#include <string.h>
#include <tvm/runtime/object.h>
#include <tvm/node/reflection.h>
#include <tvm/node/repr_printer.h>

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

// class Test : public ObjectRef {
// public:
//     Test() {}
//     explicit Test(ObjectPtr<Object> n) : ObjectRef(n) {}
//     const TestNode* operator->() const { return static_cast<const TestNode*>(get()); }
//     TestNode* operator->() { return static_cast<TestNode*>(get_mutable()); }
// };

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TestNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* op = static_cast<const TestNode*>(ref.get());
      p->stream << "Test(";
      p->stream << "name=" << op->name<< ", ";
      p->stream << ")";
    });
}
}