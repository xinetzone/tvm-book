# 将 VM 放入 TVM：Relay Virtual Machine

Relay是一种新的程序表示方法，它实现了大量机器学习程序的表示和优化。不幸的是，通过支持更有表现力的程序集，引入了一些新的执行挑战。

Relay 的解释器可以执行完整的语言，但是有明显的限制，这使得它不适合生产部署。它被构造成执行 AST 遍历来执行程序的低效解释器。这种方法在概念上很简单，但效率很低，因为 AST 遍历严重依赖于其间接性。

在编译动态代码方面还有更多的挑战，比如动态调度（dynamic scheduling）和分配（allocation）、全动态张量形状（fully dynamic tensor shapes）和控制流（control flow）。解释器为这些提供了简单的解决方案，但没有足够引人注目或优化的。

第二种执行机制是现有的图执行程序。为了使 Relay 程序达到这个目标，将它们中的一小部分编译成旧的图格式，并在运行时执行它们。图执行器提供了快速执行体验，但只适用于非常有限的 Relay 程序子集。

另一种替代的但不是标准的方法是 Relay 的提前编译器（ahead-of-time compiler），它将 Relay 程序编译到包含提前实现的共享库中。提前编译器提供了令人信服的性能，但是很难扩展和使用，这只能通过修改代码生成和优化机制来实现。

Relay 虚拟机旨在成为平衡这些相互竞争的方法的框架，提供动态执行环境，可以通过灵活的扩展机制进行扩展、插装，并与其他方法(如提前编译)集成。

设计虚拟机是为了在部署和执行 Relay 程序时在性能和灵活性之间取得平衡，而不放弃 TVM 的好处。

在编程语言和系统中，虚拟机(VM)设计是得到充分研究的领域，已经有各种成熟的和嵌入式编程语言的虚拟机设计。

以前的语言 VM 设计都是针对传统程序的执行配置文件进行了大量的定制。传统程序操作小标量值，由大量低级指令组成。指令的数量要求指令的执行和调度非常高效。在机器学习的背景下，主要使用张量值，使用(相对)少量的高级指令。ML 程序的成本中心是在大输入上昂贵的 operator 调用，例如 GEMM 或卷积。由于 ML 程序所展示的执行概要，在标量 vm 中进行微优化的重要性大大降低。

TVM 为视觉模型提供了强有力的支持，但我们希望发展到支持更广泛的模型。图执行器能够利用输入图的完全静态特性来执行主动优化，例如完全静态分配和最佳内存重用。当引入使用控制流、递归、动态形状（dynamic shapes）和动态分配（dynamic allocation）的模型时，必须改变执行的工作方式。Relay 的虚拟机是自然的选择。

本文档的其余部分提供了 Relay 虚拟机设计及其指令集的高级概述。

## 设计

VM 的设计注重简单性，而不牺牲性能。为了实现这一点，着重设计了张量 VM，而不是标量 VM。

在张量 VM 设置中，优化了对象的廉价“allocation”(通过尝试避免真正的 allocation)、静态片段（static fragments）的重用和动态形状(即 jagged tensors)的能力。

### 指令集

指令集和指令（instruction）表示的选择是虚拟机最关键的设计决策。指令的当前表示形式是 tagged union，包含 op-code 和 data payload。

重要的设计决策是指令的抽象级别（RISC vs. CISC）以及它们如何获取数据（固定宽度指令编码（fixed-width instruction encoding） vs.可变长度指令编码（variable-length encoding））。当前版本更接近 CISC，使用了像 AllocTensor 这样的复杂指令，并且由于在指令中包含了形状，所以是可变长度的。当前的指令集是非常高级的，大致对应于 Relay 中的高级运算。

### Ret

**参数**：
  
- RegName dst
- RegName result

将 ``result`` 寄存器（register）中的对象返回给调用者的 ``dst`` 寄存器。

### InvokePacked

**参数**：

- Index packed_index
- Index arity
- Index output_size
- RegName* packed_args

调用 ``packed_index`` 所表示的打包函数。``arity`` 和 ``output_size`` 用来告诉虚拟机预期有多少输入和输出。``packed_args`` 存储参数寄存器（register）列表。注意 ``Index`` 是 ``int64_t`` 的别名，它也会在其他指令中使用。

### AllocTensor

**参数**：

- RegName dst
- RegName storage
- uint32_t ndim
- int64_t* shape
- DLDataType dtype

从给定的存储块（storage block）``storage`` 中分配使用常量 shape（存储在 ``shape`` 中）和 ``dtype`` 的张量值。结果保存在 register ``dst``。

### AllocTensorReg

**参数**：

- RegName dst
- RegName storage
- RegName shape_register
- DLDataType dtype

从给定的存储块（存储在 ``storage``）中分配适当形状的张量值（存储在 ``shape_register`` 中）和 ``dtype``。结果保存在 register ``dst``。

### AllocStorage

**参数**：

- RegName dst
- RegName size
- RegName alignment
- DLDataType dtype_hint

使用给定的 ``size``、``alignment``，数据类型和 ``dtype_hint`` 分配存储块。分配的存储块存储在寄存器 ``dst`` 中。

### AllocADT

**参数**：

- RegName dst
- Index tag
- Index num_fields
- RegName* datatype_fields

使用来自注册表（register）``datatype_fields`` 的 ``num_fields`` 条目分配带有标记 ``tag`` 的数据类型。结果保存在 register ``dst``。

### AllocClosure

**参数**：

- RegName dst
- Index clo_index
- Index num_freevar
- RegName* free_vars;

分配闭包，将 VMFunction 在 ``clo_index`` 处作为其代码，并从 ``free_vars`` 寄存器中分配 ``num_freevar`` 条目。结果保存在 register ``dst``。

### GetField

**参数**：

- RegName dst
- RegName object
- Index field_index

用索引 ``field_index`` 从 ``object`` 获取字段值。结果保存在 register ``dst``。

### If

**参数**：

- RegName test
- RegName target
- Index true_offset
- Index false_offset

检查寄存器 ``test`` 处的对象是否等于 ``target``。如果相等，则相对跳转为 ``true_offset``，否则相对跳转为 ``false_offset``。

### GetTag

**参数**：

- RegName object
- RegName dst

在寄存器 ``object`` 中获取 ADT 对象的对象 tag。结果保存在 register ``dst``。

### Fatal

虚拟机执行失败。

### Goto

**参数**：

- Index pc_offset

通过 ``pc_offset`` 进行相对无条件跳转。

### Invoke

**参数**：

- Index func_index

在 ``func_index`` 调用函数，消耗 VMFunction 的 arity 字段中包含的参数数量。

### InvokeClosure

**参数**：

- RegName closure
- Index num_closure_args
- RegName* closure_args

调用 ``closure``，消耗闭包的 VMFunction 中声明的参数数量。

### LoadConst

**参数**：

- RegName dst
- Index const_index

从常量池中加载常量 ``const_index``。结果保存在 register ``dst``。

### LoadConsti

**参数**：

- Index val
- RegName dst

加载常量整数 ``val`` 来注册 ``dst``。结果是 0-rank 阶张量。

## Object 表示

利用对象协议来表示 VM 使用的对象。

目前，有三种类型的对象 ``NDArray``、``ADT`` 和 ``Closure`` 对象，分别用来表示张量、元组/列表和闭包数据。它们的详细信息可以分别在 [include/tvm/runtime/ndarray.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/ndarray.h)、[include/tvm/runtime/vm/vm.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/vm/vm.h) 和 [include/tvm/runtime/container.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/container.h) 中找到。

## Stack and State

Relay VM 维护了 stack frame，其中包含关于如何恢复前一个调用的信息。在连续空间（虚拟 register 文件）中为每个函数分配 register。

跟踪已调用的一组 Relay 函数，一个指向其字节码的指针，一个指向字节码的偏移量（称为程序计数器（program counter））。

```c
struct VirtualMachine {
  ...
  std::vector<VMFrame> frames;
  ...
  // Current function.
  size_t func_index;
  // Pointer into the current function's instructions.
  const Instruction* code;
  // Current program counter relative to the code pointer.
  size_t pc;
  ...
};
```

## Dispatch Loop

VM 的关键部分是 dispatch loop。dispatch loop 通常支配虚拟机的执行时间，但通过实验发现，对于 Relay 来说并非如此。刚刚实现了简单的 ``switch``/``goto`` dispatch loop，它基于 instruction op code 进行 dispatch。

此循环由 ``VirtualMachine::Run()`` 实现。

## VM Compiler

这个基础结构的一个重要部分是编译器，从 Relay 的完整 IR 到 bytecode 序列。VM 编译器会将 ``tvm::relay::Module`` 变换为 ``tvm::relay::vm::Executable``。可执行文件包含一组编译函数，编译函数包含在 ``tvm::relay::vm::Function`` 中。函数包含关于函数的元数据以及已编译的字节码。然后，发出的可执行对象可以通过 ``tvm::relay::vm::VirtualMachine`` 对象加载和运行。有关数据结构的完整定义，请参见 [include/tvm/runtime/vm/executable.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/vm/executable.h) 和 [include/tvm/runtime/vm/vm.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/vm/vm.h)。

## 优化

VM 编译器需要相当多的优化。它们中的每一个都被实现为由 Relay pass manager 管理的 pass。

标记为 `TODO` 的优化还没有实现。

- A-Normal Form
- Lambda Lift (see [src/relay/vm/lambda_lift.cc](https://github.com/apache/tvm/blob/main/src/relay/backend/vm/lambda_lift.cc))
- Inline Primitives (see [src/relay/vm/inline_primitives.cc](https://github.com/apache/tvm/blob/main/src/relay/backend/vm/inline_primitives.cc))
- Constant Pool Layout (see [src/relay/backend/vm/compiler.cc](https://github.com/apache/tvm/blob/main/src/relay/backend/vm/compiler.cc))
- Tail Call Optimization (TODO)
- Liveness Analysis (TODO)

## 序列化

序列化和反序列化由 Relay VM 编译器生成的可执行文件是必须的，因为可能想要将模型保存到磁盘上，然后执行推断。之前，Relay 已经在 json 文件中为图执行器生成了序列化的形式。但是，同样的格式并不直接适用于 VM，因为它会发出字节码，而不是图风格的程序。可执行文件的序列化本质上需要处理模型特定的（如 weights 和 kernels）和 VM 相关的（如字节码和全局函数名）数据。

对于 kernels，可以方便地利用现有的 TVM infra 来保存和加载已编译的库模块。这里只关注以二进制格式序列化其他几个组件，这些二进制格式按照以下部分的顺序组织。

- 全局部分：包含虚拟机使用的全局变量（函数名）。
- Constant 部分：用于存储虚拟机的 constant pool（即模型的 weights）。
- 原语名称部分：为了适应将由虚拟机调用的原语算子名称列表（即以 ``fused_`` 开头的名称）引入。原语名称被用作在编译后的 kernel 库中查找函数指针的符号。
- Code 部分：VM 函数，包括字节码，都在这个部分。dispatching 循环遍历本部分以获取执行指令。

因此，与包含权重（`.params`）、graph json （.json）和编译的 kernel 库（`.so`）的图执行器工件不同，序列化的可执行工件是由 Relay 对象文件（`.ro`）和编译的 kernel 库（`.so`）组成的。

实现了 ``save`` 函数来将可执行文件存储到磁盘并将其序列化为上述格式。同时，使用 ``load_exec`` 函数加载序列化的 kernel 二进制和可执行的相关二进制代码，这些代码将再次用于实例化 VM 对象。更多示例请参考 [test_vm_serialization.py](https://github.com/apache/tvm/blob/main/tests/python/relay/test_vm_serialization.py)。

## 未解决的问题

### 如何处理动态形状？

动态形状支持是 TVM 升级 Relay，TVM 编译器正在进行的工作。关于动态形状支持的最新更新，建议在 TVM 的讨论[论坛](https://discuss.tvm.apache.org/)中进行追踪更新。

### 如何修改 VM 以支持 JIT 编译某些代码路径？

在代码生成空间中，仍然有许多需要分析的权衡，VM 被设计得非常灵活，所以可以为未来的实验修改它。

### 如何支持异构执行？

异构执行应该是开箱即用的，假设已经注解了适当的设备副本。为了正确地做到这一点，需要运行设备 annotation 和 copying passes。