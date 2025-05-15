# MSC 计算图

```{toctree}
intro
codegen
```

MSC架构设计用于解决多个系统协同优化的问题，技术实现是使用编译器思想，基于统一的中间描述在不同系统中构建计算图。同时MSC将计算逻辑和压缩算法分开，通过对压缩算法的解耦搭建通用的模型压缩平台。

Parser + MSCGraph + Codegen构成了MSC中信息在不同框架之间传递的通路，核心部分是MSCGraph作为计算信息的载体，MSCGraph的设计类似常见的DAG类型的IR格式，包括表示结算节点的MSCJoint和表示数据的MSCTensor。
