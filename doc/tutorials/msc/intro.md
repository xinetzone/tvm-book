# MSC 简介

[MSC](https://discuss.tvm.apache.org/t/rfc-unity-msc-introduction-to-multi-system-compiler/15251)（Multi-System Compiler，多系统编译器）旨在将 `tvm` 与其他机器学习框架（例如 `torch`、`tensorflow`、`tensorrt` 等）和系统（例如训练系统、部署系统等）连接起来。借助 MSC，可以开发模型压缩方法，如高级 PTQ（训练后量化）、QAT（量化感知训练）、修剪训练、稀疏训练、知识蒸馏等。此外，MSC 将模型编译过程管理为流水线，因此可以轻松地基于 MSC 构建模型编译服务（Saas）和编译工具链（tool-chain）。

MSC 被用作 NIO.Inc 的 AI 引擎的重要部分。介绍可以在 TVMConf 2023(TVM @ NIO) 找到。

这个开源版本的 MSC 与 NIO.Inc 中的MSC有以下不同：

- NIO中的运行时优化和量化不会被包含在这个开源版本中。
- 这个版本使用relax和relay来构建MSCGraph，而在 NIO 中只使用 relay。
- 这个版本专注于自动压缩和训练相关的优化方法，而 NIO 的 AI 引擎更注重运行时加速和与自动驾驶相关的量化。

## 动机

随着 TVM 的优化，模型性能和内存管理已经达到了一个相对较高的水平。为了将模型性能提升到更高的层次，同时确保准确性，需要新的方法。模型压缩技术被证明在提高模型性能的同时减少内存消耗方面是有用的。常规的压缩方法，如修剪和量化，需要算法、软件和硬件系统的合作，这使得压缩策略难以开发和维护，因为信息格式因系统而异，压缩策略也因案例而异。为了与不同的系统协作并开发作为模型无关工具的压缩算法，需要用于保存、传递和转换信息的架构。

## 指南级解释

### MSCGraph

MSCGraph 是 MSC 的核心，它对编译器的作用类似于 IR（中间表示）。MSCGraph 是 `Relax.Function/Relay.Function` 的 DAG（有向无环图）格式。它可以在 Relax/Relay 之间转换。构建 MSCGraph 的目标是使压缩算法的开发和权重管理（这在训练时很重要）更加容易。如果所选的运行时目标不支持所有的 Calls，那么一个 Relax/Relay 模块将拥有多个 MSCGraphs。

```python
from tvm.contrib.msc.core.ir import graph_translator

# build msc graph from relax
graph = graph_translator.from_relax(mod, params, entry_name)
print(graph)

# this will export serialization file for load the graph
graph.export("graph", params=params)

# this will export prototxt file for visualize
graph.visualize("graph.prototxt")

# build msc graph to relax
module = graph_translator.to_relax(graph, params)
assert_same(mod[entry_name], module["main"])
```

```{topic} MSCGraph 和 Relex 之间的差异
- MSCGraph 具有 DAG（有向无环图）格式，而 Relax 具有表达式格式。
- MSCGraph 将张量分类为输入和权重，而 Relex 将张量定义为变量和常量。
- MSCGraph 使用节点名称（如 `conv1`，`layer1.conv1` ...）作为搜索节点的主要 ID，而 Relax 使用带有前缀的索引（如 `lvXX`，`gv`）。
```

### RuntimeManager

RuntimeManager 将 MSCGraph(s) 与不同的框架连接起来，它封装了一些常用的方法并管理 MSCTools。

```python
from tvm.contrib.msc.core.transform import msc_transform
from tvm.contrib.msc.core.runtime import create_runtime_manager
from tvm.contrib.msc.core.tools import create_tool, MSC_TOOL

# build runtime manager from module and mscgraphs
optimized_mod, msc_graph, msc_config = msc_transform(mod, params)
rt_manager = create_runtime_manager(optimized_mod, params, msc_config)
rt_manager.create_tool(MSC_TOOL.QUANTIZE, quantize_config)
quantizer = rt_manager.get_tool(MSC_TOOL.QUANTIZE)

rt_manager.load_model()
# calibrate the datas with float model
while not quantizer.calibrated:
    for datas in calibrate_datas:
        rt_manager.run(datas)
    quantizer.calibrate()
quantizer.save_strategy(strategy_file)

# load again the quantized model, without loading the weights
rt_manager.load_model(reuse_weights=True)
outputs = rt_manager.run(sample_datas)
```

### MSCTools

MSCTools 与 MSCGraph 协同工作，它们决定压缩策略并控制压缩过程。MSCTools 由 RuntimeManager 管理。

```python
from tvm.contrib.msc.core.transform import msc_transform
from tvm.contrib.msc.core.runtime import create_runtime_manager
from tvm.contrib.msc.core.tools import create_tool, MSC_TOOL

# build runtime manager from module and mscgraphs
optimized_mod, msc_graph, msc_config = msc_transform(mod, params)
rt_manager = create_runtime_manager(optimized_mod, params, msc_config)

# pruner is used for prune the model
rt_manager.create_tool(MSC_TOOL.PRUNE, prune_config)

# quantizer is used to do the calibration and quantize the model
rt_manager.create_tool(MSC_TOOL.QUANTIZE, quantize_config)

# collecter is used to collect the datas of each computational node
rt_manager.create_tool(MSC_TOOL.COLLECT, collect_config)

# distiller is used to do the knowledge distilliation
rt_manager.create_tool(MSC_TOOL.DISTILL, distill_config)
```

### MSCProcessor

MSCProcessor 为编译过程构建流水线。一个编译过程可能包括不同的阶段，每个阶段都有特殊的配置和策略。为了使编译过程易于管理，创建了 MSCProcessor。

```python
from tvm.contrib.msc.pipeline import create_msc_processor

# get the torch model and config
model = get_torch_model()
config = get_msc_config()
processor = create_msc_processor(model, config)

if mode == "deploy":
    processor.compile()
    processor.export()
elif mode == "optimize":
    model = processor.optimize()
    for ep in EPOCHS:
        for datas in training_datas:
            train_model(model)
    processor.update_weights(get_weights(model))
    processor.compile()
    processor.export()
```

配置可以从文件中加载，从而可以控制、记录和重放编译过程。这对于构建编译服务和平台至关重要。

```
{
  "workspace": "msc_workspace",
  "verbose": "runtime",
  "log_file": "MSC_LOG",
  "baseline": {
    "check_config": {
      "atol": 0.05
    }
  },
  "quantize": {
    "strategy_file": "msc_quantize.json",
    "target": "tensorrt",
  },
  "profile": {
    "repeat": 1000
  },
  ...
}
```

### MSCGym

MSCGym 是 MSC 中自动压缩的平台。它的作用类似于 AutoTVM，但其架构更像 OpenAI-Gym。MSCGym 从压缩过程中提取任务，然后利用代理和环境之间的交互来为每个任务找到最佳行动。要使用 MSCGym 进行自动压缩，请为工具设置 `gym` 配置：

```
{
      ...
      "quantize": {
        "strategy_file": "msc_quantize.json",
        "target": "tensorrt",
        “gym”:[
          {
            “record”:”searched_config.json”,
            “env”:{
              “strategy”:”distill_loss”
            },
            “agent”:{
              “type”:”grid_search”,
            }
          },
        ]
      },
      ...
}
```

## 参考级解释

MSC 中的编译流水线如下所示：

![](images/msc.jpeg)

### 核心概念：

MSCGraph：MSC 的核心 IR（中间表示）。MSCGraph 是 Relax.Function/Relay.Function 的 DAG（有向无环图）格式。

- MSC codegen：为框架生成模型构建代码（包括控制 MSCTool 的包装器）。
- RuntimeManager：管理运行时、MSCGraphs 和 MSCTools 的抽象模块。
- MSCTools：决定压缩策略并控制压缩过程的工具。此外，还为调试添加了一些额外的工具到MSCTools中。
- Config：MSC 使用配置来控制编译过程。这使得编译过程易于被记录和重放。

### 编译过程：

编译过程包含两个主要阶段：优化(optimize)和最终化(finalize)。optimize and finalize

优化阶段用于通过压缩来优化模型。这个阶段可能会使用训练框架，并且消耗大量的时间和资源（例如自动压缩、知识蒸馏和训练）。

最终化阶段用于在所需环境中构建模型。这个阶段从优化后的 `relax` 模块（检查点）开始，并在目标环境中构建该模块，不进行任何优化。这个阶段可以在所需环境中进行处理，而不会消耗大量时间和资源。
