# PyTorch 环境配置指南

## CUDA 和 cuDNN 环境配置

设置以下环境变量以支持 CUDA 和 cuDNN：

```bash
export PATH=/usr/local/nvidia/bin:${PATH}
export PATH=/usr/local/cuda/bin:${PATH}
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}
export C_INCLUDE_PATH=/usr/local/cuda/include:${C_INCLUDE_PATH}
export LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:${LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:${LD_LIBRARY_PATH}
# Ensure the local libcuda have higher priority than the /usr/local/cuda/compact
# since the compact libcuda does not work on non-Tesla gpus
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH}
```

## 软件包安装

执行以下命令安装必要的软件包：

```bash
conda install conda-forge::backtrace
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install cudnn cuda=12.8
conda install -c conda-forge gcc=12.4.0 # CUDA 需要 gcc<13
python -m invoke config --cuda
python -m invoke make
```

## Vulkan 支持配置

参考文档：
- [NVIDIA Vulkan 官方文档](https://developer.nvidia.cn/vulkan)

```bash
sudo apt install libvulkan1 spirv-tools spirv-headers
conda install conda-forge::vulkan-tools
```
