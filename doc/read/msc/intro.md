# MSC 简介

参考：[【我与TVM二三事 后篇（1）】MSC简介](https://zhuanlan.zhihu.com/p/680638069)

[MSC](https://discuss.tvm.apache.org/t/rfc-unity-msc-introduction-to-multi-system-compiler/15251)（Multi-System Compiler，多系统编译器）旨在将 `tvm` 与其他机器学习框架（例如 `torch`、`tensorflow`、`tensorrt` 等）和系统（例如训练系统、部署系统等）连接起来。借助 MSC，可以开发模型压缩方法，如高级 PTQ（训练后量化）、QAT（量化感知训练）、修剪训练、稀疏训练、知识蒸馏等。此外，MSC 将模型编译过程管理为流水线，因此可以轻松地基于 MSC 构建模型编译服务（Saas）和编译工具链（tool-chain）。

MSC 中的编译流水线如下所示：

![](images/msc.jpeg)

## 核心概念

MSCGraph：MSC 的核心 IR（中间表示）。MSCGraph 是 Relax.Function/Relay.Function 的 DAG（有向无环图）格式。

- MSC codegen：为框架生成模型构建代码（包括控制 MSCTool 的包装器）。
- RuntimeManager：管理运行时、MSCGraphs 和 MSCTools 的抽象模块。
- MSCTools：决定压缩策略并控制压缩过程的工具。此外，还为调试添加了一些额外的工具到MSCTools中。
- Config：MSC 使用配置来控制编译过程。这使得编译过程易于被记录和重放。
