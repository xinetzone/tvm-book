# 表达式算子

算子是神经网络模型的构建模块。深度神经网络可以表示为有向无环图（Directed Acyclic Graph，简称 DAG），其中节点是算子，边是节点之间的数据依赖关系。对于高性能的神经网络模型 execution 来说，能够高效地执行算子是必不可少的。

在 {ref}`ch_vector_add_te` 中，您已经看到了如何在 TVM 中构建向量加法表达式。本章涵盖了 TVM 中构造表达式的更多概念。具体来说，您将学习数据类型、形状、索引、reduction 和控制流，在下一章中，您将能够基于这些内容构造算子。

```{toctree}
data_types
shapes
index_shape_expressions
reductions
if_then_else
all_any
```
