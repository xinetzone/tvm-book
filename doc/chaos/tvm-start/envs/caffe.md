# Caffe 环境配置指南

```{warning}
Caffe 环境配置有许多坑，请自行抉择，其实，若仅仅使用 caffe 做推理，完全不需要完整的 caffe 环境，详情见 [caffeproto](../../../frontend/relax/caffe/caffeproto)
```

## 安装最小依赖库

本指南介绍在 Linux（Ubuntu/Debian 系统）上安装最小依赖库来运行 CPU 模式的 Caffe。

```{note} 
conda 与系统依赖包对应关系

| 依赖库 | 对应系统包 |
|--------|------------|
| protobuf | libprotobuf-dev + protobuf-compiler |
| leveldb | libleveldb-dev |
| snappy | libsnappy-dev |
| opencv | libopencv-dev |
| hdf5 | libhdf5-serial-dev |
| boost | libboost-all-dev |
| openblas | atlas（BLAS）替代 |
| gflags | libgflags-dev |
| glog | libgoogle-glog-dev |
| lmdb | liblmdb-dev |
```

这些依赖库是 Caffe 运行的基础组件：

| 依赖库 | 功能 |
|--------|------|
| Protocol Buffers | 序列化数据结构 |
| LevelDB/LMDB | 数据存储后端 |
| Snappy | 数据压缩 |
| OpenCV | 图像处理 |
| HDF5 | 数据存储格式 |
| Boost | C++ 工具库 |
| BLAS（OpenBLAS） | 基础线性代数运算 |
| GFlags | 命令行参数解析 |
| GLog | 日志记录 |

2. 获取源码

```bash
git clone git@github.com:xinetzone/caffe.git
cd caffe
```

3. 配置 `Makefile.config`

```bash
cp Makefile.config.example Makefile.config

# 开启 CPU_ONLY，关闭所有 GPU/CUDA 相关：
sed -i 's/# CPU_ONLY := 1/CPU_ONLY := 1/' Makefile.config
```

````{admonition} 关闭 MLSL 支持
:class: note

MLSL (Machine Learning Scaling Library) 是 Intel 提供的分布式训练加速库。对于单机训练场景，可以安全关闭此功能：

```bash
# 禁用 MLSL 支持
sed -i 's/USE_MLSL := 1/USE_MLSL := 0/' Makefile.config
```

关闭 MLSL 的优势：
| 优势 | 说明 |
|------|------|
| 依赖简化 | 移除对 MLSL 库的依赖 |
| 编译加速 | 减少编译时间和复杂度 |
| 资源节省 | 降低运行时内存占用 |
| 单机优化 | 专为单机训练场景优化，无需分布式计算支持 |
````

4. 编译和测试

```bash
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  -DCPU_ONLY=ON \
  -DUSE_MLSL=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DBUILD_python=OFF \
  -DUSE_OPENCV=OFF \
  -DUSE_LEVELDB=OFF \
  -DUSE_LMDB=OFF \
  -DUSE_MKLDNN_AS_DEFAULT_ENGINE=OFF \
  -DUSE_OPENMP=OFF \
  -DDISABLE_ABSL=OFF \
  -DBUILD_python_layer=OFF \
  -DBUILD_docs=OFF \
  -DCMAKE_CXX_STANDARD=14 \
  -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DMKL2017_SUPPORTED=OFF \
  -DMKLDNN_SUPPORTED=OFF \
  -DBUILD_TESTING=OFF \
  -DBLAS=open
make all -j$(nproc)
# 测试
make test -j$(nproc)
make runtest -j$(nproc)
```

5. 安装到系统或自定义路径（可选）

```bash
# 安装到系统目录
sudo make install
# 或安装到自定义目录（如 $HOME/caffe_install）
make install PREFIX=$HOME/caffe_install
```
