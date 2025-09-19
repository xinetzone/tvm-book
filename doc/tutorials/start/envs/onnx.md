# ONNX 环境配置指南

ONNX 生态系统包含多个组件，建议安装以下核心包：

```bash
# 基础安装
pip install onnx onnxscript -i https://pypi.tuna.tsinghua.edu.cn/simple

# GPU加速支持 (需要CUDA环境)
pip install onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 安装验证

运行以下 Python 代码验证安装是否成功：

```python
import onnx
import onnxruntime

print(f"ONNX版本: {onnx.__version__}")
print(f"ONNX Runtime版本: {onnxruntime.__version__}")

# 检查GPU支持
print(f"ONNX Runtime GPU可用: {onnxruntime.get_device() == 'GPU'}")
```
