# MSC 简介

MSC 中的编译流水线如下所示：

![](images/msc.jpeg)

## 核心概念

MSCGraph：MSC 的核心 IR（中间表示）。MSCGraph 是 Relax.Function/Relay.Function 的 DAG（有向无环图）格式。

- MSC codegen：为框架生成模型构建代码（包括控制 MSCTool 的包装器）。
- RuntimeManager：管理运行时、MSCGraphs 和 MSCTools 的抽象模块。
- MSCTools：决定压缩策略并控制压缩过程的工具。此外，还为调试添加了一些额外的工具到MSCTools中。
- Config：MSC 使用配置来控制编译过程。这使得编译过程易于被记录和重放。
