#ifndef TVM_CONTRIB_MSC_UTILS_PLUGIN_BASE_H_
#define TVM_CONTRIB_MSC_UTILS_PLUGIN_BASE_H_

#include <cassert>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace tvm {
namespace contrib {
namespace msc {
namespace plugin {

typedef enum {
  kUINT8 = 0,
  kINT8 = 1,
  kINT16 = 2,
  kINT32 = 3,
  kINT64 = 4,
  kFLOAT16 = 5,
  kFLOAT32 = 6,
  kFLOAT64 = 7,
  kUNKNOWN = 8,
} MetaDataType;

class MetaShape {
 public:
  MetaShape() { shape_.resize(0); }

  MetaShape(const std::vector<int64_t>& shape) {
    for (auto d : shape) {
      shape_.push_back(d);
    }
  }

  template <typename T>
  void SetShape(const std::vector<T>& shape) {
    for (auto d : shape) {
      shape_.push_back(static_cast<int64_t>(d));
    }
  }

  template <typename T>
  void SetDim(int index, T dim) {
    int valid_index = index < 0 ? shape_.size() + index : index;
    if (valid_index >= shape_.size()) {
      std::string err =
          std::to_string(index) + " out of dims size " + std::to_string(shape_.size());
      throw std::runtime_error(err);
    }
    shape_[valid_index] = dim;
  }

  template <typename T>
  const std::vector<T> GetShape() const {
    std::vector<T> shape;
    for (auto d : shape_) {
      shape.push_back(d);
    }
    return shape;
  }

  inline int64_t DimAt(int index) const {
    int valid_index = index < 0 ? shape_.size() + index : index;
    if (valid_index >= shape_.size()) {
      std::string err =
          std::to_string(index) + " out of dims size " + std::to_string(shape_.size());
      throw std::runtime_error(err);
    }
    return shape_[valid_index];
  }

  inline size_t ndim() const { return shape_.size(); }

  inline const std::vector<int64_t> shape() const { return shape_; }

  inline size_t size() const {
    size_t size = 1;
    for (auto d : shape_) {
      assert(d > 0 && "Can not compute static size with unknow dim");
      size *= d;
    }
    return size;
  }

  inline int64_t operator[](int index) const { return DimAt(index); }

  friend std::ostream& operator<<(std::ostream& out, const MetaShape& shape) {
    for (size_t i = 0; i < shape.ndim(); i++) {
      out << shape.DimAt(i) << (1 < shape.ndim() ? "" : ",");
    }
    return out;
  }

 private:
  std::vector<int64_t> shape_;
};

class MetaLayoutAxis {
 public:
  MetaLayoutAxis(const char name, size_t factor = 0) : factor_(factor) {
    name_ = (factor == 0 ? "" : std::to_string(factor)) + std::string(1, name);
  }

  MetaLayoutAxis(const std::string& name) {
    if (name.size() == 1) {
      factor_ = 0;
      name_ = name;
    } else {
      factor_ = std::stoi(name.substr(1));
      name_ = name.substr(0, 1);
    }
  }

  inline const std::string name() const { return name_; }

  inline size_t factor() const { return factor_; }

 private:
  std::string name_;
  size_t factor_;
};

class MetaLayout {
 public:
  MetaLayout() {}

  MetaLayout(const std::string& name) : name_(name) {
    int factor = 0;
    for (char c : name) {
      if (c >= 'A' && c <= 'Z') {
        assert(factor == 0 && "Upper layout axis do not accept factor");
        MetaLayoutAxis axis(c);
        axes_.push_back(axis);
      } else if (c >= 'a' && c <= 'z') {
        assert(factor > 0 && "Lower layout axis should has factor");
        MetaLayoutAxis axis(c, factor);
        axes_.push_back(axis);
        factor = 0;
      } else if (c >= '0' && c <= '9') {
        assert(factor >= 0 && "Factor number should between 0 and 9");
        factor = factor * 10 + c - '0';
      } else {
        throw std::runtime_error("Unexpected layout axis " + name);
      }
    }
    CheckValid();
  }

  MetaLayout(const std::vector<MetaLayoutAxis>& axes) : axes_(axes) {
    name_ = "";
    for (auto a : axes_) {
      name_ += (a.factor() == 0 ? "" : std::to_string(a.factor())) + a.name();
    }
    CheckValid();
  };

  void CheckValid() {
    std::set<std::string> recorded_axes;
    for (auto a : axes_) {
      auto axis_name = a.name();
      assert(!recorded_axes.count(axis_name) && ("Has duplicate layout axis in " + name_).c_str());
      recorded_axes.insert(axis_name);
    }
  }

  inline const MetaLayoutAxis AxisAt(int index) const {
    int valid_index = index < 0 ? axes_.size() + index : index;
    if (valid_index >= axes_.size()) {
      std::string err = std::to_string(index) + " out of axes size " + std::to_string(axes_.size());
      throw std::runtime_error(err);
    }
    return axes_[valid_index];
  }

  inline MetaLayoutAxis operator[](int index) { return AxisAt(index); }

  inline size_t ndim() const { return axes_.size(); }

  inline std::string name() const { return name_; }

  friend std::ostream& operator<<(std::ostream& out, const MetaLayout& layout) {
    out << layout.name();
    return out;
  }

 private:
  std::string name_;
  std::vector<MetaLayoutAxis> axes_;
};

class MetaTensor {
 public:
  MetaTensor() {}

  MetaTensor(const MetaShape& shape, const MetaDataType& data_type,
             const MetaLayout& layout = MetaLayout())
      : shape_(shape), data_type_(data_type), layout_(layout) {}

  inline const MetaShape shape() const { return shape_; }

  inline MetaDataType data_type() const { return data_type_; }

  inline const std::vector<int64_t> meta_shape() const { return shape_.shape(); }

  inline const MetaLayout layout() const { return layout_; }

  inline const std::string layout_name() const { return layout_.name(); }

  inline size_t ndim() const { return shape_.ndim(); }

  inline size_t size(bool count_batch = true) const {
    if (count_batch) {
      size_t batch_dim = 0;
      for (size_t i = 0; i < layout_.ndim(); i++) {
        if (layout_.AxisAt(i).name() == "N") {
          batch_dim = i;
        }
      }
      return shape_.size() / shape_.shape()[batch_dim];
    }
    return shape_.size();
  }

  inline MetaLayoutAxis AxisAt(int index) const { return layout_.AxisAt(index); }

  inline int AxisOf(const std::string& axis) const {
    for (size_t i = 0; i < layout_.ndim(); i++) {
      if (layout_.AxisAt(i).name() == axis) {
        return i;
      }
    }
    return -1;
  }

  inline int64_t DimAt(int index) const { return shape_.DimAt(index); }

  inline int64_t DimAt(const std::string& axis) const {
    int idx = AxisOf(axis);
    if (idx >= 0) {
      return shape_.DimAt(idx);
    }
    throw std::runtime_error("Can not find dim for " + axis);
  }

  friend std::ostream& operator<<(std::ostream& out, const MetaTensor& tensor) {
    out << "tensor : <" << tensor.shape() << ">, (" << tensor.layout() << ")";
    return out;
  }

 private:
  MetaShape shape_;
  MetaDataType data_type_;
  MetaLayout layout_;
};

template <typename T>
class DataTensor : public MetaTensor {
 public:
  DataTensor(const MetaShape shape, const MetaDataType& data_type, const MetaLayout layout, T* data)
      : MetaTensor(shape, data_type, layout) {
    data_ = data;
  }

  DataTensor(const MetaShape shape, const MetaDataType& data_type, const MetaLayout layout,
             const T* data)
      : MetaTensor(shape, data_type, layout) {
    data_ = const_cast<T*>(data);
  }

  T* data() const { return data_; }

  const T* const_data() const { return data_; }

 private:
  T* data_{nullptr};
};

}  // namespace plugin
}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_UTILS_PLUGIN_BASE_H_
