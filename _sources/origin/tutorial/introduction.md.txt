(sphx_glr_tutorial_introduction.py)=
(sphx_glr_tutorial_introduction)=
# TVM 和模型优化的概述

下图说明了机器学习模型在用 TVM 优化编译器框架进行转换时的步骤。

![A High Level View of TVM](https://tvm.apache.org/images/tutorial/overview.png)

1. 从 *Tensorflow*、*PyTorch* 或 *Onnx* 等框架导入模型。

   importer 层是 TVM 可以从其他框架中导入模型的地方，比如 Tensorflow、PyTorch 或 ONNX。由于这个开源项目在不断改进，TVM 为每个前端提供的支持水平也不尽相同。如果你在将模型导入 TVM 时遇到问题，你可能想尝试将其转换为 ONNX。

2. 翻译成 *Relay*，TVM 的高级模型语言。

   已经导入 TVM 的模型是用 Relay 表示的。Relay 是一种函数式语言（functional language）和神经网络的中间表示法（IR）。它支持以下内容：

   - 传统的数据流式表示法
   - Functional-style scoping 和 let-binding 使其成为一种功能齐全的可区分语言
   - 能够允许用户混合两种编程风格

   Relay 应用图级（graph-level）优化通道来优化模型。

3. lower 到 *张量表达式* （Tensor Expression，简称 TE）表示。

   lower 是指高层表示被转化为低层表示。在应用高层优化后，Relay 运行 FuseOps，将模型分割成许多小的子图，并将子图 lower 到 TE 表示。

   张量表达式（TE）是用于描述张量计算的专属域语言。

   TE 还提供了几个 *schedule* 原语来指定低级的循环优化，例如平铺（tiling）、矢量化（vectorization）、并行化（parallelization）、unrolling 和 fusion。

   为了帮助将 Relay 表示转换为 TE 表示的过程，TVM 包含张量算子清单（Tensor Operator Inventory，简称 TOPI），它有预先定义的常见张量算子的模板（如 conv2d、transpose）。

4. 使用 auto-tuning 模块 *AutoTVM* 或 *AutoScheduler* 搜索最佳 schedule。

   schedule 指定在 TE 中定义了算子或子图的低级循环优化。auto-tuning 模块搜索最佳 schedule 并将其与 cost 模型和设备上的测量结果进行比较。

   在 TVM 中，有两个 auto-tuning 模块：

   - **AutoTVM**：基于模板的 auto-tuning 模块。它运行搜索算法为用户定义的模板中的可调节旋钮找到最佳值。

      对于常见的运算符，其模板已经在 TOPI 中提供。

   - **AutoScheduler** （别名 Ansor） ：无模板的自动调谐模块。

      它不需要预先定义的 schedule 模板。相反，它通过分析计算的定义自动生成搜索空间。

      然后，它在生成的搜索空间中搜索最佳 schedule。

5. 选择最佳配置进行模型编译。tuning 后，auto-tuning 模块会生成 JSON 格式的 auto-tuning 记录。这一步为每个子图挑选出最佳的 schedule。

6. lower 到张量级的中间表示（Tensor Intermediate Representation，简称 TIR），TVM 的低层次中间表示。

   在根据 tuning 步骤选择最佳配置后，每个 TE 子图被降低到 TIR，并通过低级别的优化通道进行优化。

   接下来，优化后的 TIR 被降低到硬件平台的目标编译器中。这是最后的代码生成阶段，产生可以部署到生产中的优化模型。
   
   TVM 支持几种不同的编译器后端，包括：

   - LLVM：它可以针对任意的微处理器架构，包括 标准 x86 和 ARM 处理器，AMDGPU 和 NVPTX 代码生成，以及 LLVM 支持的任何其他平台。
   - 专门的编译器，如 NVCC，NVIDIA 的编译器。
   - 嵌入式和专用目标，通过 TVM 的 Bring Your Own Codegen（BYOC）框架实现。

7. 编译成机器码。在这个过程结束时，特定的编译器生成的代码可以 lower 为机器码。

   TVM 可以将模型编译成可链接的对象模块，然后可以用轻量级的 TVM 运行时来运行，该运行时提供 C 语言的 API 来动态加载模型，以及其他语言的入口，如 Python 和 Rust。TVM 还可以建立捆绑式部署，其中运行时与模型结合在一个包中。

本教程的其余部分将更详细地介绍 TVM 的这些方面。
