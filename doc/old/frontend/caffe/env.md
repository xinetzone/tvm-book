# `caffe` 环境

`caffe` 安装的依赖很复杂，有两种便捷方法使用使用 caffe。

## 构建 `caffe` docker 环境

[`docker` 环境](https://hub.docker.com/r/xinetzone/tvmx/tags)，且此环境依赖
```bash
pip install scikit-image protobuf==3.20.3 numpy==1.26.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
细节见：[TVM docker 镜像制作](https://zhuanlan.zhihu.com/p/560671997)。

## 直接使用 docker 环境中的 `/opt/caffe/python/caffe`

把 `/opt/caffe/python/caffe` 复制到你的机器上，便可以直接作为 Python 包使用。

