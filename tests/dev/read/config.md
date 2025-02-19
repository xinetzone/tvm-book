# 项目配置

以 Ubuntu 为例说明，如何在本地模式下配置项目环境。

:::::{card-carousel} 2
::::{card} 初始化环境
1. 使用 `conda` 创建并激活名为 `bare` 的虚拟环境：
```bash
conda create -n bare python=3.10
conda activate bare
```

2. 安装任务管理工具：
```bash
pip install invoke
```
::::
::::{card} 配置项目
1. 初始化项目
```bash
invoke init
```
2. 安装项目环境
```bash
invoke install
```
::::
::::{card} 项目文档（可选）
1. 安装项目文档依赖
```bash
invoke install --group doc
```
2. 生成项目文档
```bash
invoke pdm-doc
```
3. 清理项目文档
```bash
invoke pdm-doc --cmd clean
```
:::::

## 配置 TVM 环境（ubuntu 平台）

```{rubric} 将 TVM 源码库克隆到本地，并编译生成动态库
```

```bash
git clone --recurse-submodules git@github.com:xinetzone/tvm.git
cd tvm
pip install d2py
cd xinetzone
invoke init
invoke config
invoke make
```

````{note}
如果需要启用 CUDA，可以选择：
```bash
invoke config --cuda
```
````

### TVM 环境搭建方案

下面提供两种 TVM 环境搭建方案：

:::::{card}
:shadow: md
```{rubric} 配置 TVM 临时环境
```

1. 创建文件夹 `tvm-ai/` 用于开发 TVM 应用，然后安装 `tvm-book` 工具链：

```bash
pip install tvm-book
```

2. 创建配置文件 `tvm-ai/tvm.toml`，用于配置 TVM 本机环境，写入内容如下：

```toml
TVM_ROOT = "/media/pc/data/lxw/ai/tvm"
```

```{admonition} 这样做的目的
为后续添加新的配置提供便利。
```

3. 测试环境是否正确配置：

```python
import toml

from ._env import set_tvm
from tvm_book.config.env import set_tvm

def set_env(toml_path):
    """
    Args:
        toml_path: 存储 TVM 环境配置
    """
    with open(toml_path) as fp:
        config = toml.load(fp)
    set_tvm(config["TVM_ROOT"])

set_env("tvm.toml")
import tvm
import vta
```
:::::

:::::{card}
:shadow: md
```{rubric} tvm-book 项目内开发
```

1. 直接在 `tvm-book` 项目根目录安装项目环境：

```bash
invoke init
invoke install
invoke tvm -r /media/pc/data/lxw/ai/tvm/xinetzone
```

2. 在 `tvm-book/` 项目内任意深度的目录均可随意使用 TVM/VTA。
:::::