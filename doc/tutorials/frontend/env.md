# 配置前端环境

以 Python3.11 为基础环境。

```bash
conda create -n py311 python=3.11
conda activate py311
```

## PyTorch 前端环境

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## TensorFlow 前端环境

```bash
pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
```
