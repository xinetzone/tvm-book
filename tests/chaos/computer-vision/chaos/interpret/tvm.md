# TVM 项目解读

本篇关注 TVM 资源库的根部的部分文件夹：

- `/src/`：用于算子编译和运行时部署的 C++ 代码。
- `/src/relay/`：Relay 实现。一种新深度学习框架的函数式 IR（中间表示）。它管理计算图的组件，计算图中的节点使用 `/src/` 其他实现的基础设施进行编译和执行。
- `/python`：Python 前端，包装 `/src/` 中实现的 C++ 函数和对象。为 C++ API 和 `/src/driver/` 代码提供 Python 绑定，用户可以用它来执行编译。与每个节点对应的算子在 `/src/relay/op` 中注册。
- `/src/topi`：计算标准神经网络算子的定义和后端调度。它们是用 C++ 或 Python 编码的算子实现。

当用户通过 {func}`tvm.relay.build` 调用计算图的编译时，对图中的每个节点都会发生以下一系列动作：

- 通过查询算子注册表查找算子实现
- 为算子生成计算（`compute`）表达式和调度
- 将算子编译成目标代码（object code）

```{hint}
TVM 中 Python 和 C++ 可以相互调用。
```