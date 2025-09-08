# FFI 简介

FFI（外部函数接口（Foreign Function Interface） 特性：
- 专门的干净 Any/AnyView，可以存储项的强引用和弱引用
- 基于 Any/AnyView 构建的函数（以前称为 PackedFunc）系统
- 最小的 C API，支持整体调用。目标是带来干净的、稳定的 FFI 规范，适用于编译和注册代码
- 基于模块重写的核心 Python 绑定和生成代码
- 最新 dlpack 支持

好处：
- Any 可以支持 POD 类型（int）和对象类型。
- 容器（例如 Array）现在也可以包含 Any 值，例如支持 Array<int> ，无需 boxed 类型
- 错误处理为基于对象的方式，允许跨语言更清晰的回溯
- 映射（Map）保留插入顺序
- 隔离稳定最小核心 ABI/API 基础模块路径
- 基于类型特性的设计，清晰定义值与 Any 系统如何交互
- 根据特征按需自动转换不同类型

## 升级说明
- 对于像 `Map<ObjectRef, ObjectRef>` 这样的通用容器，考虑使用 `Map<Any, Any>` 代替
- 用 `ffi::Function` 代替 `PackedFunc`
- 现在 `cast<T>` 或 `.as<T>()` （返回可选）需要显式转换以实现更好的类型安全
    - 可能可以插入 `args[i].cast<T>()` 来显式转换为 `T`，或者使用带类型的版本
- 对于需要某种形式 boxing 的地方，属性大多变成 POD 类型，例如现在使用 `bool` 和 `int64_t` 来表示 `int` 和 `bool` 属性

## {mod}`tvm.ffi` 模块

{mod}`tvm.ffi` 模块实现 TVM C 语言核心 API 与 Python 的接口绑定，是 Python 调用底层 C++ 实现的关键桥梁。可以单独使用不依赖其他组件，这种设计便于：
- 模块化部署
- 减少依赖冲突
- 快速集成到其他系统
