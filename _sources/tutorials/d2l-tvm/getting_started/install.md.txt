(ch_install)=
# 安装

## 安装运行环境

本项目使用 Anaconda3 配置 Python 环境，还需要配置一些包：

- {mod}`tvm_book` 和 {mod}`mxnet`
- 深度学习编译器 {mod}`tvm.topi`（{cite:p}`Chen.Moreau.Jiang.ea.2018`）

```{attention}
由于 TVM 高度依赖于原生系统的可用库，推荐从其源码安装它。详细内容可参考：[从源码安装 TVM](https://daobook.github.io/tvm/docs/install/from_source.html)。
```

本书需要一些 TVM 编译配置：

```bash
set(USE_LLVM ON)
```

如果您计划在 Nvidia GPU 上运行，那么您也需要

```bash
set(USE_CUDA ON)
```

也不要忘记启用 `cython`，它可以加速性能。您只需要在 TVM 源文件夹中运行 `make cython`。

安装 MXNet GPU 版本（{cite:p}`Chen.Li.Li.ea.2015`）可以：

```bash
pip install mxnet-cu112
```

## 便捷安装

```{attention}
当前仅仅支持 Unubtu 平台。
```

可以直接利用脚本 {download}`task.py <https://github.com/daobook/tvm/blob/xin/xinetzone/tasks.py>` （需要安装 `d2py` 和 `invoke`）编译 TVM。

1. 构建 `build/` 目录，且创建并配置 `build/cmake/config.cmake` 文件。

    ```bash
    invoke config
    ```

    若要启用 CUDA，可以：

    ```bash
    invoke config --cuda
    ```

2. 构建并编译 TVM：

    ```bash
    invoke make
    ```

3. 若要生成 TVM 文档，可：

    ```bash
    invoke update
    invoke doc
    ```

4. 如果想要把 {mod}`tvm` 和 {mod}`vta` 作为全局环境的包，可以在 `tvm` 项目下创建软链接：

    ```bash
    invoke ln-env --root tvm项目根目录 --target python环境路径 --python-version 3.10
    ```

## 关于 `tvm_book` 库

为了便于本项目程序的复用，定制了 `tvm_book` 库，可以直接在项目根目录直接安装：

```bash
pip install .
```

如若想要本地化项目文档，可以

```bash
pip install .[doc]
```

如若不想将 `tvm_book` 安装到全局环境，可以在本次运行代码前添加如下函数用于设置 Python 临时环境：

```python
def set_env(num, current_path='.'):
    '''
    num 表示相对于 current_path 的父级根目录级别
    '''
    import sys
    from pathlib import Path

    ROOT = Path(current_path).resolve().parents[num]
    sys.path.extend([str(ROOT/'src')]) # 设置 `tvm_book` 环境
    from tvm_book.tvm.env import set_tvm 
    # 设置 TVM 环境
    set_tvm(TVM_ROOT)
```

```{note}
`set_tvm` 需要自行配置以适配您的设备。
```

## 安装的 FAQs

安装 MXNet GPU 版本可能需要配置 NCCL，详细内容见：[安装 NCCL](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)。

```{note}
这里是 Ubuntu22.04，可以替换为其他版本。
```

1. 安装 CUDA：

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```

2. 安装 NCCL：

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
```

同时还需要：

```bash
sudo apt install libnccl2=2.13.4-1+cuda11.7 libnccl-dev=2.13.4-1+cuda11.7
```
