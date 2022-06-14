# VTA

VTA（发音为 vita，全称 versatile tensor accelerator，Open, Modular, Deep Learning Accelerator Stack）是一个开源的深度学习加速器，它与基于端到端虚拟机的编译器栈相辅相成。具有如下特性：

- 通用、模块化（modular）、开源硬件
    - 简化部署到 FPGA 的工作流程。
    - 仿真器支持在常规工作站上 prototype compilation passes。
- 仿真器和 FPGA 硬件后端驱动程序和 JIT 运行时。
- 端到端 TVM 堆栈集成
    - 通过 TVM 直接优化和部署深度学习框架的模型。
    - 可定制和扩展的 TVM 编译器后端。
    - 灵活的 RPC 支持，以简化部署，并使用 Python 以方便编程 FPGA。

```{toctree}
:hidden:

config
test
```