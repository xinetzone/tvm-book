/*! TVM 插件
 * \brief 
 * \file nd_sub_class.cc
 */
// #include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
// #include <tvm/tir/op.h>

using namespace tvm;
// using namespace tvm::tir;
using namespace tvm::runtime;
namespace tvm_ext {
/*!
 * \brief TVM 的 NDArray 的子类。
 *
 * 要使用此插件，外部库应执行以下操作：
 *
 * 1) 继承 TVM 的 NDArray 和 NDArray 容器；
 *
 * 2) 遵循新的对象协议以将新 NDArray 定义为引用类。
 *
 * 3) 在 Python 前端上，继承 `tvm.nd.NDArray`，
 *    并使用 `tvm.register_object` 注册类型。
 */
class NDSubClass : public tvm::runtime::NDArray {
 public:
  class SubContainer : public NDArray::Container {
   public:
    SubContainer(int additional_info) : additional_info_(additional_info) {
      type_index_ = SubContainer::RuntimeTypeIndex();
    }
    int additional_info_{0};

    static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
    static constexpr const char* _type_key = "tvm_ext.NDSubClass";
    TVM_DECLARE_FINAL_OBJECT_INFO(SubContainer, NDArray::Container);
  };

  static void SubContainerDeleter(Object* obj) {
    auto* ptr = static_cast<SubContainer*>(obj);
    delete ptr;
  }

  NDSubClass() {}
  explicit NDSubClass(ObjectPtr<Object> n) : NDArray(n) {}
  explicit NDSubClass(int additional_info) {
    SubContainer* ptr = new SubContainer(additional_info);
    ptr->SetDeleter(SubContainerDeleter);
    data_ = GetObjectPtr<Object>(ptr);
  }

  NDSubClass AddWith(const NDSubClass& other) const {
    SubContainer* a = static_cast<SubContainer*>(get_mutable());
    SubContainer* b = static_cast<SubContainer*>(other.get_mutable());
    ICHECK(a != nullptr && b != nullptr);
    return NDSubClass(a->additional_info_ + b->additional_info_);
  }
  int get_additional_info() const {
    SubContainer* self = static_cast<SubContainer*>(get_mutable());
    ICHECK(self != nullptr);
    return self->additional_info_;
  }
  using ContainerType = SubContainer;
};
TVM_REGISTER_OBJECT_TYPE(NDSubClass::SubContainer);

/*!
 * \brief Introduce additional extension data structures
 *        by sub-classing TVM's object system.
 */
class IntVectorObj : public Object {
 public:
  std::vector<int> vec;

  static constexpr const char* _type_key = "tvm_ext.IntVector";
  TVM_DECLARE_FINAL_OBJECT_INFO(IntVectorObj, Object);
};

/*!
 * \brief Int vector reference class.
 */
class IntVector : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(IntVector, ObjectRef, IntVectorObj);
};

TVM_REGISTER_OBJECT_TYPE(IntVectorObj);
} // namespace tvm_ext

namespace tvm_ext {
TVM_REGISTER_GLOBAL("tvm_ext.ivec_create").set_body([](TVMArgs args, TVMRetValue* rv) {
  auto n = tvm::runtime::make_object<IntVectorObj>();
  for (int i = 0; i < args.size(); ++i) {
    n->vec.push_back(args[i].operator int());
  }
  *rv = IntVector(n);
});

TVM_REGISTER_GLOBAL("tvm_ext.ivec_get").set_body([](TVMArgs args, TVMRetValue* rv) {
  IntVector p = args[0];
  *rv = p->vec[args[1].operator int()];
}); 

TVM_REGISTER_GLOBAL("tvm_ext.nd_create").set_body([](TVMArgs args, TVMRetValue* rv) {
  int additional_info = args[0];
  *rv = NDSubClass(additional_info);
  ICHECK_EQ(rv->type_code(), kTVMNDArrayHandle);
});

TVM_REGISTER_GLOBAL("tvm_ext.nd_add_two").set_body([](TVMArgs args, TVMRetValue* rv) {
  NDSubClass a = args[0];
  NDSubClass b = args[1];
  *rv = a.AddWith(b);
});

TVM_REGISTER_GLOBAL("tvm_ext.nd_get_additional_info").set_body([](TVMArgs args, TVMRetValue* rv) {
  NDSubClass a = args[0];
  *rv = a.get_additional_info();
});

} // namespace tvm_ext