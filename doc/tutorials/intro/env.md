# TVM 环境配置

配置工具：

```bash
conda create -n py312 python=3.12
conda activate py312
cd tvm/xinetzone
pip install .[doc,dev] -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pickleshare -i https://pypi.tuna.tsinghua.edu.cn/simple # 用于 jupyter notebook 魔法指令
```

配置环境：

```bash
python -m invoke init
python -m invoke config --cuda
python -m invoke make
```


配置前端框架(包含 CUDA+CUDNN)：

```bash
conda install pytorch torchvision torchaudio cuda-nvcc cudnn  cudatoolkit pytorch-cuda=12.4 -c pytorch -c nvidia
pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install onnx onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
# caffe 依赖
pip install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install protobuf==3.20.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
