# 配置

关于内核配置的详细说明见：[项目启动](https://xinetzone.github.io/sphinx-demo/start/index.html)。

## 安装 TVM

前提条件：

1. 安装好 VSCode、Anaconda3

### 配置环境

- 克隆源码库：

    ```sh
    git clone https://github.com.cnpmjs.org/daobook/tvm.git
    cd tvm
    ```

- 设置 Git 子库（国内下载速度慢的一种解决方案）：

    ```sh
    git submodule init
    ```

    修改 .git/config 子模块的地址为：

    ```
    [submodule "3rdparty/cutlass"]
	url = https://github.com.cnpmjs.org/NVIDIA/cutlass
    [submodule "dlpack"]
        url = https://github.com.cnpmjs.org/dmlc/dlpack
    [submodule "dmlc-core"]
        url = https://github.com.cnpmjs.org/dmlc/dmlc-core
    [submodule "3rdparty/libbacktrace"]
        url = https://github.com.cnpmjs.org/tlc-pack/libbacktrace.git
    [submodule "3rdparty/rang"]
        url = https://github.com.cnpmjs.org/agauniyal/rang
    [submodule "3rdparty/vta-hw"]
        url = https://github.com.cnpmjs.org/apache/incubator-tvm-vta
    ```

    执行 `update` 即可：

    ```sh
    git submodule update
    ```

### Windows10+ 安装 TVM

- 使用 [`mamba`](https://github.com/mamba-org/mamba) 提升 `conda` 速度：

    ```sh
    conda install -c conda-forge mamba
    ```

- 创建 TVM：

    ```sh
    mamba env create --file conda/build-environment.yaml
    ```

### Ubuntu 从源码创建 TVM

1. 准备最小的先决条件：

    ```sh
    sudo apt-get update
    sudo apt-get install -y gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
    ```

2. 自己创建一个环境，然后安装并配置 LLVM：

    ```sh
    conda create -n tvmx python=3.8
    conda activate tvmx
    ```

    可以使用 `conda` 安装：

    ```sh
    conda install -c anaconda llvm
    ```

    或者 `apt` 安装：

    ```sh
    sudo apt install llvm
    ```

3. 构建目标库 `libtvm.so` 和 `libtvm_runtime.so`：

    ```sh
    mkdir build
    cp cmake/config.cmake build
    cd build
    cmake ..
    make -j8
    ```

更多关于安装和配置 TVM 的内容见：[从源码安装](install-from-source)