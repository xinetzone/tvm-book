
.. _deploy-and-integration:

部署模型并集成到 TVM
===============================

本页面是如何将 TVM 部署到各种平台以及如何将其与您的项目集成的指南。

.. image::  https://tvm.apache.org/images/release/tvm_flexible.png

构建 TVM 运行时库
-----------------------------

.. _build-tvm-runtime-on-target-device:

与传统的深度学习框架不同。TVM 堆栈分为两个主要组件：

- TVM 编译器（compiler）：完成模型的所有编译和优化
- TVM 运行时（runtime）：在目标设备上运行

为了集成已编译的模块，**不需要** 在目标设备上构建整个 TVM。
您只需要在 desktop 上构建 TVM 编译器堆栈，并使用它来交叉编译部署在目标设备上的模块。

只需要使用轻量级的 runtime API，它可以集成到各种平台中。

例如，在基于 Linux 的嵌入式系统（如 Raspberry Pi）上，可以通过以下命令构建运行时 API：

.. code:: bash

    git clone --recursive https://github.com/apache/tvm tvm
    cd tvm
    mkdir build
    cp cmake/config.cmake build
    cd build
    cmake ..
    make runtime

注意，输入 ``make runtime`` 来只构建运行时库。

还可以交叉编译运行时。运行时库的交叉编译不应与嵌入式设备的交叉编译模型混淆。

如果你想包含额外的运行时，如 OpenCL，你可以修改 ``config.cmake`` 来启用这些选项。
在获得 TVM 运行时库之后，您可以链接已编译的库

.. figure:: https://raw.githubusercontent.com/tlc-pack/web-data/main/images/dev/tvm_deploy_crosscompile.svg
   :align: center
   :width: 85%

模型（TVM 优化或未优化）可以由 TVM 针对不同的架构进行交叉编译，例如在 ``x64_64`` host 上的 ``aarch64``。
一旦模型被交叉编译，为了能够运行交叉编译的模型，就必须有与目标架构兼容的运行时。

交叉编译其他架构的 TVM 运行时
-----------------------------------------------------

在 :ref:`上面的 <build-tvm-runtime-on-target-device>` 例子中，运行时库是在树莓派上编译的。
与树莓派（Raspberry Pi）等目标设备相比，在拥有高性能处理器和充足资源的主机（即 host，如笔记本电脑、工作站）上生成运行时库的速度要快得多。
为了交叉编译运行时，必须安装目标设备的工具链（toolchain）。
在安装了正确的工具链之后，与原生编译的主要区别是向 cmake 传递一些额外的命令行参数，以指定要使用的工具链。
作为参照，在现代笔记本电脑（使用 8 个线程）上为 ``aarch64`` 构建 TVM 运行时库大约需要 20 秒，而在树莓派4 上构建运行时需要 10 分钟。

aarch64 的交叉编译
"""""""""""""""""""""""""

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

.. code-block:: bash

    cmake .. \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_VERSION=1 \
        -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc \
        -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ \
        -DCMAKE_FIND_ROOT_PATH=/usr/aarch64-linux-gnu \
        -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
        -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
        -DMACHINE_NAME=aarch64-linux-gnu 

    make -j$(nproc) runtime

对于 bare metal ARM 设备，可以使用以下工具链代替 gcc-aarch64-linux-* 进行安装

.. code-block:: bash

   sudo apt-get install gcc-multilib-arm-linux-gnueabihf g++-multilib-arm-linux-gnueabihf


RISC-V 的交叉编译
"""""""""""""""""""""""""

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu


.. code-block:: bash

    cmake .. \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_VERSION=1 \
        -DCMAKE_C_COMPILER=/usr/bin/riscv64-linux-gnu-gcc \
        -DCMAKE_CXX_COMPILER=/usr/bin/riscv64-linux-gnu-g++ \
        -DCMAKE_FIND_ROOT_PATH=/usr/riscv64-linux-gnu \
        -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
        -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
        -DMACHINE_NAME=riscv64-linux-gnu 

    make -j$(nproc) runtime


``file`` 命令可用于查询生成的运行时的体系结构。

.. code-block:: bash

   file libtvm_runtime.so
   libtvm_runtime.so: ELF 64-bit LSB shared object, UCB RISC-V, version 1 (GNU/Linux), dynamically linked, BuildID[sha1]=e9ak845b3d7f2c126dab53632aea8e012d89477e, not stripped

    
针对目标设备优化和调优模型
-------------------------------------------

在嵌入式设备上测试、调优和 benchmark TVM kernel 的最简单和推荐的方法是通过 TVM 的 RPC API。相关教程的链接如下：

- :ref:`tutorial-cross-compilation-and-rpc`
- :ref:`tutorial-deploy-model-on-rasp`

在目标设备上部署已优化的模型
----------------------------------------

在完成调优和基准测试之后，可能需要在目标设备上部署模型，而不需要依赖 RPC。关于如何这样做，请参阅以下参考资料。

.. toctree::
   :maxdepth: 2

   cpp_deploy
   android
   adreno
   integrate
   hls
   arm_compute_lib
   tensorrt
   vitis_ai
   bnns

额外的部署指南
-----------------------------

还开发了一些针对特定设备的操作指南，可以在 Jupyter 笔记本中查看可用的 Python 代码。这些“如何操作”描述了如何准备模型并将其部署到许多受支持的后端。

- :ref:`sphx_glr_how_to_deploy_models`
