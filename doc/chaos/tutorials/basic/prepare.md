# TVM 准备

(ch_install)=
## 安装运行环境

本项目使用 Anaconda3 配置 Python 环境，还需要配置一些包：

- {mod}`tvm_book` 和 {mod}`d2py`
- 深度学习编译器 {mod}`tvm`（{cite:p}`Chen.Moreau.Jiang.ea.2018`）

```{attention}
由于 TVM 高度依赖于原生系统的可用库，推荐从其源码安装它。详细内容可参考：[从源码安装 TVM](https://xinetzone.github.io/tvm/docs/install/from_source.html)。
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

（可选）安装 MXNet GPU 版本（{cite:p}`Chen.Li.Li.ea.2015`）可以：

```bash
pip install mxnet-cu116
```

（可选）安装 PyTorch 可以：

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

（可选）安装 ONNX，可以：

```bash
pip install onnx onnxoptimizer 
```

## 便捷安装

```{attention}
当前仅仅支持 Unubtu 平台。
```

可以直接利用脚本 {download}`task.py <https://github.com/xinetzone/tvm/blob/xin/xinetzone/tasks.py>` （需要安装 `d2py` 和 `invoke`）编译 TVM。

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
````{note}
如若想更新 TVM 文档翻译只需要
```bash
invoke doc.update
invoke doc.update -l zh_CN
```
````

4. 如果想要把 {mod}`tvm` 和 {mod}`vta` 作为全局环境的包，可以在 `tvm` 项目下创建软链接：

    ```bash
    invoke ln-env --root tvm项目根目录 --target python环境路径 --python-version 3.10
    ```

5. 如果 TVM 和 VTA 仅仅是作为局部使用，可以使用 `pdm` 进行管理。

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
    from tvm_book.config.env import set_tvm
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

TVM 安装可以是：

`````{tab-set}
````{tab-item} CPU
```bash
pip install mlc-ai-nightly -f https://mlc.ai/wheels
```
````
````{tab-item} GPU
```bash
pip install mlc-ai-nightly-cu110 -f https://mlc.ai/wheels
```
````
`````

## 自动调优

为了在 TVM 中使用 `autotvm` 包，需要安装一些额外的依赖项。


```bash
pip install psutil cloudpickle
```

还需要安装 `xgboost`：

`````{tab-set}
````{tab-item} CPU
```bash
pip install xgboost 
```
````
````{tab-item} GPU
```bash
conda install -c conda-forge py-xgboost-gpu
```
````
`````

## 查看模型

查看模型的 input/shape_dict 的推荐方法是通过 [netron](https://netron.app/)。打开模型后，单击第一个节点，在 inputs 部分查看名称和形状。

## 其他

加速 `scikit-learn` 可以安装 [scikit-learn-intelex](https://intel.github.io/scikit-learn-intelex)：

```bash
conda install scikit-learn-intelex
```
