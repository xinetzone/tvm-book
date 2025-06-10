# TVM 环境配置

鉴于需要保证 TVM 项目的独立性，本项目将作为 TVM 的插件存在，需要先配置 TVM 环境。

1. 本项目默认采用 `python=3.12`（其他版本，可以自行调试）

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

2. 在本项目之外将 [TVM 项目](https://github.com/xinetzone/tvm) 克隆下来，并安装一些初始化包：

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

3. （可选）配置前端框架(包含 CUDA+CUDNN)：

    ```bash
    # Environment variables
    export PATH=/usr/local/nvidia/bin:${PATH}
    export PATH=/usr/local/cuda/bin:${PATH}
    export CPLUS_INCLUDE_PATH=/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}
    export C_INCLUDE_PATH=/usr/local/cuda/include:${C_INCLUDE_PATH}
    export LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:${LIBRARY_PATH}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:${LD_LIBRARY_PATH}

    # Ensure the local libcuda have higher priority than the /usr/local/cuda/compact
    # since the compact libcuda does not work on non-Tesla gpus
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH}
    conda install conda-forge::backtrace
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -i https://pypi.tuna.tsinghua.edu.cn/simple
    conda install cudnn
    pip install tensorflow onnx onnxscript onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
    conda install -c conda-forge gcc=12.4.0 # CUDA 需要 gcc<13
    python -m invoke config --cuda
    python -m invoke make
    ```

````{note}
可以直接在 CPU 上测试 TVM 功能

```bash
%%shell
# Installs the latest dev build of TVM from PyPI. If you wish to build
# from source, see https://tvm.apache.org/docs/install/from_source.html
pip install apache-tvm --pre
```
````

## Vulkan 支持

文档：
- [Nvidia Vulkan](https://developer.nvidia.cn/vulkan)

```bash
sudo apt install libvulkan1 spirv-tools spirv-headers
conda install conda-forge::vulkan-tools
```

## conda 安装

可以使用 conda 安装 TVM 环境：
```bash
conda env create -f tvm-conda.yaml
```
