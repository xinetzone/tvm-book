# TVM 基础环境配置指南

为确保 TVM 项目的独立性，本项目作为 TVM 的插件运行，请先完成以下 TVM 环境配置：

## 1. 基础环境配置

默认使用 Python 3.13 版本，如需其他版本请自行调整：

```bash
# 确保从全新的环境开始
conda env remove -n ai
# 使用构建依赖创建 conda 环境
conda create -n ai -c conda-forge \
    "llvmdev>=15" \
    "cmake>=3.24" \
    "gcc<=12.4.0" \
    git \
    python=3.13
# 进入构建环境
conda activate ai
```

## 2. TVM 项目初始化

在项目目录外克隆 [TVM 项目](https://github.com/xinetzone/tvm) 并安装必要的初始化包：

```bash
git clone --recursive https://github.com/xinetzone/tvm
cd tvm/xinetzone
python -m pip install --upgrade pip
pip install taolib[flows]
invoke init
invoke config
invoke make
pip install -e .[dev] -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 3. Conda 安装方式

替代方案：通过 conda 安装 TVM 环境：
```bash
conda env create -f tvm-conda.yaml
```

## 4. 常见问题解决

1. 若在 TVM 编译环境中遇到 `GLIBCXX_3.4.30' not found` 错误，请执行以下安装：

```bash
# 通过 conda 安装
conda install -c conda-forge libstdcxx-ng=14.2

# 验证库版本
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX
```

2. CUDA_nvToolsExt_LIBRARY-NOTFOUND

```bash
conda install nvidia::cuda-nvtx
```