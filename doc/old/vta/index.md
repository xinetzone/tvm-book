# VTA：开源、模块化的深度学习加速器栈

VTA（通用张量加速器）是一款开源深度学习加速器，配有基于 TVM 的端到端编译器栈。

VTA 的主要特性包括：

- 通用、模块化、开源硬件
  - 简化的 FPGA 部署工作流程
  - 仿真器支持，可在普通工作站上原型化编译流程
- 适用于仿真器和 FPGA 硬件后端的驱动程序和 JIT 运行时
- 端到端 TVM 栈集成
  - 通过 TVM 直接优化和部署来自深度学习框架的模型
  - 可定制和可扩展的 TVM 编译器后端
  - 灵活的 RPC 支持，便于部署，并能通过 Python 便捷地对 FPGA 进行编程
        
```{toctree}
environment
intro
vta-insn
tutorials/index
benchmark/index
```
