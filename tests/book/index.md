# 文档构建

```bash
inv doc.build --opts "-b docx" -t ./doc/_build/docx
```

目录
第一篇 TVM 技术
第一章	TVM模型
1.1 TVM 概述
1.1.1 TVM 环境搭建
1.1.2 Conda 环境搭建
1.1.3 TVM 环境定制
1.2 PackedFunc
1.2.1注册全局函数
1.2.2 Python 调用 C++ 接口
1.2.3 convert 函数
1.2.4 Numpy 与 TVM 互转
1.2.5 PyTorch 与 TVM 互转
1.3 使用张量表达式处理算子
		1.3.1 编写和调度 TE 中的向量加法
		1.3.2 用 TE 手动优化矩阵乘法
	1.4  tvm.tir.trace 追踪 TE
		1.4.1 获取 TE 中间状态
		1.4.2 回调外部函数
	1.5 Relay 表达式表示神经网络
		1.5.1 自定义分类网络
		1.5.2 自定义检测网络
1.6 使用 TEDD 进行可视化
1.7 可视化 Relay 模块

第二章 前端模型
2.1 Pytorch 模型
		2.1.1 PyTorch 模型概述
		2.1.2 TorchScript 模型
			2.1.2.1 torch.jit.trace 模型
			2.1.2.2 torch.jit.script 模型
		2.1.3 PyTorch trace模型变换为 Relay 模型
		2.1.4 PyTorch script 模型变换为 Relay 模型
		2.1.5 PTQ 模型变换为 Relay 模型
		2.1.6 QAT 模型变换为 Relay 模型
2.2 TensorFlow 模型
		2.2.1 Tensorflow1 前端
			2.2.1.1 Tensorflow(pb) 转 ONNX
			2.2.1.2 Tensorflow 前端之 TF-slim
			2.2.1.3 TensorFlow1 pb 推理
			2.2.1.4 TensorFlow1 Relay 推理
2.2.3 TensorFlow2 前端
	2.2.3.1 升级 TF1 为 TF2
	2.2.3.2 TensorFlow2 推理
	2.2.3.3 TensorFlow2 Relay 推理
2.2.7 TensorFlow2 Keras 推理
2.3 ONNX 模型
		2.3.1 ONNX 模型概述
		2.3.2 ONNX Runtime 概述
		2.3.3 ONNX Script
			2.3.3.1 ONNX Script 简介
			2.3.3.2 ONNX Script 基础特性
			2.3.3.3 ONNX 自定义算子
			2.3.3.4 ONNX Script 急切模式推理
			2.3.3.5 ONNX 导出 LibProto
			2.3.3.6 ONNX ModelProto 存储模型属性
			2.3.3.7 ONNX Script 模型本地函数
			2.3.3.8 ONNX Script 模式重写
			2.3.3.9 ONNX Script 模型优化
			2.3.3.10 扩展 ONNX 注册表
		2.3.4 PyTorch 模型转换为ONNX 模型
			2.3.4.1 MobileNetV3 变换为 ONNX 模型
			2.3.4.2 YOLOV8 变换为 ONNX 模型
		2.3.5 ONNX 模型变换为 Relay 模型
2.4 Caffe 模型
		2.3.1 Caffe 模型概述
		2.3.2 Caffe 模型变换为 Relay 模型
2.5 OneFlow 模型
		2.4.1 OneFlow 模型概述
		2.4.2 OneFlow 模型变换为 Relay 模型
2.6 Paddlepaddle 模型
		2.5.1 Paddlepaddle 模型概述
		2.5.2 Paddlepaddle 模型变换为 Relay 模型
第三章 Relay 模型简化
3.1	数据流模式
3.1.1 模板匹配
3.1.2 模式分割
3.1.3 重写模式
3.1.4 融合模式
3.1.5 模式化简
3.2 自定义 Relay算子
3.2.1构建 reshape4d_softmax_reshape2d ONNX 算子
3.2.2 变换 reshape4d_softmax_reshape2d 为 softmax_transpose_reshape2d
3.2.3 定义并注册softmax_transpose_reshape2d 算子
3.2.4 定义 softmax_transpose_reshape2d 计算与调度
3.2.5 验证 softmax_transpose_reshape2d 数值一致性
第四章 Relay 模型量化
4.1 量化配置与预热
	4.1.1 量化配置
	4.1.2 常量折叠
4.1.3 算子融合
4.2 量化分区
4.4 量化注解
4.5 量化策略
	4.5.1 全局量化
	4.5.2 校准量化
4.6 量化实现
	4.6.1 定点乘法
	4.6.2 浮点模型转换为定点模型
4.7 自定义 Relay量化算子

第二篇 VTA 技术
第五章 VTA 教程
   6.1 VTA 安装与配置
   6.2 VTA 简单的矩阵乘法
   6.3 VTA 编译编译深度学习模型
	6.3.1 编译分类模型
	6.3.2 编译检测模型
	6.3.3 GNN 模型
   6.4 VTA 优化 Tensor算子
	6.4.1 分块矩阵乘法
	6.4.2 2D 卷积计算优化
   6.5 自动调优
第三篇 MLC 技术
第六章 Relax 模型
7.1  TensorIR张量程序抽象
7.1.1 元张量函数
	7.1.2 TensorIR 案例研究
7.2 端到端模型执行
	7.2.1 ONNX 模型转换为 Relax 模型
7.2.2 PyTorch 模型转换为 Relax 模型
7.2.3 利用 tvm.relax.frontend,nn 构建模型
7.3 自动程序优化
7.4 集成机器学习框架
7.5 GPU 加速
7.6 计算图优化
7.7 Relay 模型变换为 Relax 模型

第七章 模型构建与部署
   8.1 构建 TVM 模型
	8.1.1 TVM 模型构建
	8.1.2 Relay 模型构建
   8.2 获取目标源码
   8.3 Python 端部署
   8.4 C++ 端部署

附录与资源
数学符号表
软件与工具库
进一步阅读与研究资源


