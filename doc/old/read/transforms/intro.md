# 变换概述

`tvm.ir.transform` 定义了 IR 变体之间的通用传递基础设施。

{class}`tvm.ir.transform.ModulePass` 是在 `tvm.IRModule` 上工作的 pass。用户不需要直接与该类交互。相反，应该通过 `module_pass(pass_func=None, opt_level=None, name=None, required=None, traceable=False)` 创建模块级传递，因为 `module_pass` API 的设计足够灵活，可以以不同的方式处理模块级传递的创建。此外，可以从基类访问模块 pass 的所有成员。

{class}`tvm.ir.transform.Sequential` 处理 pass 对象序列的传递。可以使用这个类顺序地执行多个传递。请注意，用户还可以提供一系列在运行顺序传递时不希望应用的传递。pass 依赖项也将在后端进行解析。

````{tab-set}
```{tab-item} PassInfo
{class}`tvm.ir.transform.PassInfo` 类包含 pass 所需的元数据。它是运行优化或分析所需信息的容器。当需要更多元数据时，可以通过添加新成员来扩展这个类。

- `name` （`str`）是 pass 名称
- `opt_level` （``int``） 表示在哪个优化级别将启用传递
- `required` （`list[str]`） 表示执行某个传递所需的依赖
```
```{tab-item} Pass
所有 Pass 的基类。这里的所有方法都只是在后端实现的简单包装器。它们的定义是为了方便用户与基类进行交互。
```
```{tab-item} PassContext
{class}`tvm.ir.transform.PassContext` 表示 Relay 优化/分析运行的基础。

每个 pass 上下文都包含许多辅助信息，用于帮助优化 pass。这些信息包括记录优化过程中误差的误差报告器等。

`opt_level` 
:   `int | None`

    The optimization level of this pass.

`required_pass`
:   `list[str] | set[str] | tuple[str] | None`

    The list of passes that are required by a certain pass.

`disabled_pass`
:   `list[str] | set[str] | tuple[str] | None`

    The list of passes that are disabled.

`instruments`
:   `Sequence[PassInstrument] | None`

    The list of pass instrument implementations.

`config`
:   `dict[str, Object] | None`

    Additional configurations for specific passes.

`trace`
:   `relax.tuning.Trace | None`

    Initial trace for trace mode.

`trace_stack`
:   `list[relax.tuning_api.Trace] | None`

    Initial trace stack for trace mode.

`make_traceable`
:   `list[str] | None`

    List of passes to make traceable.

`num_evals`
:   `int`

    initial number of evaluations conducted in the pipeline.

`tuning_api_database`
:   `relax.tuning_api.JSONDatabase | None`
```
````