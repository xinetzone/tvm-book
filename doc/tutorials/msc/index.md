# MSC

[MSC](https://discuss.tvm.apache.org/t/rfc-unity-msc-introduction-to-multi-system-compiler/15251)（Multi-System Compiler，多系统编译器）旨在将 `tvm` 与其他机器学习框架（例如 `torch`、`tensorflow`、`tensorrt` 等）和系统（例如训练系统、部署系统等）连接起来。借助 MSC，可以开发模型压缩方法，如高级 PTQ（训练后量化）、QAT（量化感知训练）、修剪训练、稀疏训练、知识蒸馏等。此外，MSC 将模型编译过程管理为流水线，因此可以轻松地基于 MSC 构建模型编译服务（Saas）和编译工具链（tool-chain）。

