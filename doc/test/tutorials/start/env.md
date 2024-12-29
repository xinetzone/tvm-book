# TVM 环境配置

鉴于需要保证 TVM 项目的独立性，本项目将作为 TVM 的插件存在，需要先配置 TVM 环境。

1. 考虑到 Python 版本的隔离，本项目默认采用 `python=3.12`（其他版本，可以自行调试）

    ```bash
    conda create -n ai python=3.12
    conda activate ai
    ```

2. 在本项目之外将 [TVM 项目](https://github.com/xinetzone/tvm) 克隆下来，并安装一些初始化包：

    ```bash
    git clone https://github.com/xinetzone/tvm
    cd tvm/xinetzone
    pip install .[doc,dev] -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -m invoke init
    ```

3. 编译 TVM 运行时，以备后续使用：

    ```bash
    python -m invoke config
    python -m invoke make
    ```

4. （可选）配置前端框架(包含 CUDA+CUDNN)：

    ```bash
    conda install conda-forge::backtrace
    conda install cuda cudnn pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
    pip install tensorflow onnx onnxscript onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -m invoke config --cuda
    python -m invoke make
    ```

````{note}
`caffe` 安装的依赖很复杂，暂时没有更好的办法，推荐使用 [`docker` 环境](https://hub.docker.com/r/xinetzone/tvmx/tags)，且此环境依赖
```bash
pip install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install protobuf==3.20.3 numpy==1.26.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
细节见：[TVM docker 镜像制作](https://zhuanlan.zhihu.com/p/560671997)。

鉴于精力有限，`caffe` 环境的维护将不再继续。
````

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
conda install conda-forge::vulkan-tools
```
