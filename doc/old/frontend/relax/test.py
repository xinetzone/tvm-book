import tvm
from tvm import relax, te
import numpy as np

# 创建一个示例张量
def create_example_tensor():
    # 创建一个形状为 (1, 3, 4, 4) 的随机浮点数张量
    # 模拟一个批次大小为1、3个通道、4x4特征图的卷积输出
    np.random.seed(42)  # 设置随机种子，确保结果可复现
    data_np = np.random.uniform(-5.0, 5.0, (1, 3, 4, 4)).astype(np.float32)
    return tvm.nd.array(data_np, dtype="float32")

# 基本量化和反量化示例
def basic_quantize_dequantize_example():
    print("=== 基本量化和反量化示例 ===")
    
    # 创建输入张量
    data = create_example_tensor()
    print(f"原始数据形状: {data.shape}")
    print(f"原始数据类型: {data.dtype}")
    print(f"原始数据前几个值: {data[0, 0, 0, :4]}")
    
    # 设置量化参数
    # 简单起见，这里使用标量缩放因子和零点
    scale = tvm.nd.array(np.array([0.01]), dtype="float32")  # 缩放因子
    zero_point = tvm.nd.array(np.array([0]), dtype="int8")    # 零点
    
    # 执行量化操作 (float32 -> int8)
    quantized_data = relax.op.quantize(
        data,
        scale=scale,
        zero_point=zero_point,
        axis=-1,  # 这里使用最后一维作为量化轴，但由于使用标量参数，实际会应用于所有元素
        out_dtype="int8"
    )
    
    # 执行反量化操作 (int8 -> float32)
    dequantized_data = relax.op.dequantize(
        quantized_data,
        scale=scale,
        zero_point=zero_point,
        axis=-1,
        out_dtype="float32"
    )
    
    # 将 TVM 表达式转换为可执行函数
    # 在实际应用中，通常会将这些操作构建到完整的计算图中
    f = relax.build(
        relax.Function([], [quantized_data, dequantized_data], [relax.TensorStructInfo(), relax.TensorStructInfo()]),
        target="llvm",
    )
    
    # 创建运行时并执行
    vm = relax.VirtualMachine(f, tvm.cpu())
    quant_result, dequant_result = vm()
    
    print(f"量化后数据类型: {quant_result.dtype}")
    print(f"量化后数据前几个值: {quant_result[0, 0, 0, :4]}")
    print(f"反量化后数据类型: {dequant_result.dtype}")
    print(f"反量化后数据前几个值: {dequant_result[0, 0, 0, :4]}")
    
    # 计算量化误差
    error = np.abs(data.asnumpy() - dequant_result.asnumpy())
    print(f"量化-反量化最大绝对误差: {np.max(error)}")
    print(f"量化-反量化平均绝对误差: {np.mean(error)}")

# 通道级量化示例
def per_channel_quantize_example():
    print("\n=== 通道级量化示例 ===")
    
    # 创建输入张量
    data = create_example_tensor()
    print(f"原始数据形状: {data.shape}")
    
    # 为每个通道设置不同的缩放因子和零点
    # 假设我们有3个通道
    num_channels = data.shape[1]  # 第二个维度是通道维度
    
    # 为每个通道计算合适的缩放因子（简单起见，这里手动设置）
    scales = tvm.nd.array(np.array([0.01, 0.02, 0.03]), dtype="float32")
    zero_points = tvm.nd.array(np.array([0, 1, 2]), dtype="int8")
    
    print(f"每个通道的缩放因子: {scales.asnumpy()}")
    print(f"每个通道的零点: {zero_points.asnumpy()}")
    
    # 执行通道级量化操作
    # 注意 axis=1 表示在通道维度上应用不同的量化参数
    quantized_data = relax.op.quantize(
        data,
        scale=scales,
        zero_point=zero_points,
        axis=1,  # 在通道维度上应用不同的量化参数
        out_dtype="int8"
    )
    
    # 执行通道级反量化操作
    dequantized_data = relax.op.dequantize(
        quantized_data,
        scale=scales,
        zero_point=zero_points,
        axis=1,
        out_dtype="float32"
    )
    
    # 将 TVM 表达式转换为可执行函数
    f = relax.build(
        relax.Function([], [quantized_data, dequantized_data], [relax.TensorStructInfo(), relax.TensorStructInfo()]),
        target="llvm",
    )
    
    # 创建运行时并执行
    vm = relax.VirtualMachine(f, tvm.cpu())
    quant_result, dequant_result = vm()
    
    # 计算每个通道的量化误差
    for c in range(num_channels):
        channel_data = data.asnumpy()[:, c:c+1, :, :]
        channel_dequant = dequant_result.asnumpy()[:, c:c+1, :, :]
        error = np.abs(channel_data - channel_dequant)
        print(f"通道 {c} 的最大量化误差: {np.max(error)}")

# 不同量化类型示例
def different_quant_types_example():
    print("\n=== 不同量化类型示例 ===")
    
    # 创建一个简单的输入张量
    data_np = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    data = tvm.nd.array(data_np)
    print(f"原始数据: {data.asnumpy()}")
    
    # 共享的量化参数
    scale = tvm.nd.array(np.array([0.1]), dtype="float32")
    
    # 测试不同的输出整数类型
    quant_types = ["int8", "uint8", "int16"]
    
    for dtype in quant_types:
        # 为不同类型选择合适的零点
        if dtype == "int8":
            zero_point = tvm.nd.array(np.array([0]), dtype="int8")
        elif dtype == "uint8":
            zero_point = tvm.nd.array(np.array([128]), dtype="uint8")  # 对于uint8，通常选择128作为零点
        elif dtype == "int16":
            zero_point = tvm.nd.array(np.array([0]), dtype="int16")
        
        # 执行量化和反量化
        quantized = relax.op.quantize(data, scale, zero_point, axis=-1, out_dtype=dtype)
        dequantized = relax.op.dequantize(quantized, scale, zero_point, axis=-1, out_dtype="float32")
        
        # 编译和执行
        f = relax.build(
            relax.Function([], [quantized, dequantized], [relax.TensorStructInfo(), relax.TensorStructInfo()]),
            target="llvm",
        )
        vm = relax.VirtualMachine(f, tvm.cpu())
        quant_result, dequant_result = vm()
        
        print(f"\n量化类型: {dtype}")
        print(f"量化后数据: {quant_result.asnumpy()}")
        print(f"反量化后数据: {dequant_result.asnumpy()}")
        print(f"量化误差: {np.abs(data_np - dequant_result.asnumpy())}")

# 运行所有示例
if __name__ == "__main__":
    basic_quantize_dequantize_example()
    per_channel_quantize_example()
    different_quant_types_example()