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
    pip install pickleshare -i https://pypi.tuna.tsinghua.edu.cn/simple # 用于 jupyter notebook 魔法指令
    python -m invoke init
    ```

3. 编译 TVM 运行时，以备后续使用：

    ```bash
    python -m invoke config --cuda
    python -m invoke make
    ```

4. （可选）配置前端框架(包含 CUDA+CUDNN)：

    ```bash
    conda install cuda-nvcc cudnn cudatoolkit pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install onnx onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
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
