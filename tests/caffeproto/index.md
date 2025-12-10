# 最小化 caffe protobuf 库

参考：[caffe-proto](https://github.com/daoflows/caffe)

创建最小化的 caffe protobuf 库文件。

## 安装依赖

1. 在 [conda 环境中安装 protobuf](https://daobook.github.io/pygallery/study/fields/protobuf/installation.html)。

```bash
conda create -n proto-env python=3.12
conda activate proto-env
conda install -c conda-forge libprotobuf
pip install protobuf --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. 安装 cmake、conan 和 ninja

```bash
conda install -c conda-forge ninja
pip install cmake conan -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 生成 Python 代码

```bash
# 检测 Conan 环境
conan profile detect --force
# 安装依赖
conan install . -c tools.cmake.cmaketoolchain:generator=Ninja --build=missing
# 配置 CMake
cmake --preset conan-release
# 构建项目
cmake --build --preset conan-release
```
