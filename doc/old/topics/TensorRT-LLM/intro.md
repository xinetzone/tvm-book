# TensorRT-LLM 简介

TensorRT-LLM 包含了所有优化（即内核融合和量化、C++ 实现等运行时优化、KV 缓存、连续飞行中批处理和分页注意力）以及更多功能，同时提供了直观的模型定义 API，用于定义和构建新模型。

TensorRT-LLM 提供的一些主要优势包括：
- [通用 LLM 支持](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html#support-matrix-software)
- [流式批处理](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html#inflight-batching)(In-Flight Batching)和分页注意力机制(Paged Attention)：飞行批处理利用了 LLM 的整体文本生成过程可以被分解为多个模型执行迭代的优势。TensorRT-LLM 运行时不会等待整个批次完成才继续下一批请求，而是立即从批次中移除已完成的序列。然后，它在其他请求仍在进行时开始执行新的请求。这是旨在减少队列等待时间、消除填充请求需求并允许更高 GPU 利用率的 [Executor API](https://nvidia.github.io/TensorRT-LLM/advanced/executor.html#executor)。
- 多 GPU 多节点推理：TensorRT-LLM 包含预处理和后处理步骤，以及多 GPU 多节点通信原语，在简单、开源的模型定义 API 中，为 GPU 上的 LLM 推理性能带来突破。更多详情请参阅[多 GPU 和多节点支持](https://nvidia.github.io/TensorRT-LLM/architecture/core-concepts.html#multi-gpu-multi-node)部分。
- FP8 支持：配备 TensorRT-LLM 的 [NVIDIA H100 GPU](https://www.nvidia.com/en-us/data-center/dgx-h100/) 可让您轻松将模型权重转换为新的 FP8 格式，并自动编译模型以利用优化的 FP8 内核。这是通过 [NVIDIA Hopper](https://blogs.nvidia.com/blog/h100-transformer-engine/) 实现的，无需更改任何模型代码。
- 最新 GPU 支持：TensorRT-LLM 支持基于 NVIDIA Hopper、NVIDIA Ada Lovelace 和 NVIDIA Ampere 架构的 GPU。可能存在某些限制。
- 原生 Windows 支持：自 v0.18.0 版本起，Windows 平台支持已被弃用。所有与 Windows 相关的代码和功能将在未来的版本中完全移除。


## 可以用 TensorRT-LLM 做什么？

让 TensorRT-LLM 加速 NVIDIA GPU 上最新 LLM 的推理性能。将 TensorRT-LLM 用作 NVIDIA NeMo 中 LLM 推理的优化核心，NeMo 是端到端的框架，用于构建、定制和部署生成式 AI 应用到生产环境。NeMo 为生成式 AI 部署提供完整的容器，包括 TensorRT-LLM 和 NVIDIA Triton。

TensorRT-LLM 通过开源的模块化模型定义 API 提高了易用性和可扩展性，用于定义、优化和执行新的架构和增强功能，随着 LLM 的演进，可以轻松定制。

## 在 Linux 上安装

安装 TensorRT-LLM（在 Ubuntu 24.04 上测试）：

```bash
(Optional) pip3 install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

sudo apt-get -y install libopenmpi-dev && pip3 install --upgrade pip setuptools && pip3 install tensorrt_llm
```

需要 PyTorch CUDA 12.8 软件包来支持 NVIDIA Blackwell GPU。在之前的 GPU 上，不需要这个额外的安装。
```bash
conda install cuda=12.8 cudnn
```

如果使用 [PyTorch NGC 容器镜像](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)，安装支持 NVIDIA Blackwell 的 PyTorch 软件包和 `libopenmpi-dev` 的先决步骤是不需要的。

详细安装教程见：[TensorRT-LLM Linux 安装](https://nvidia.github.io/TensorRT-LLM/installation/linux.html)。
