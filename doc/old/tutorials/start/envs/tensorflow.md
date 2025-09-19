# TensorFlow 环境配置指南

安装 TensorFlow：

```bash
pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 验证安装

安装完成后，运行以下命令验证安装是否成功：

```python
import tensorflow as tf
print(tf.__version__)
print("GPU可用:", tf.config.list_physical_devices('GPU'))
```
