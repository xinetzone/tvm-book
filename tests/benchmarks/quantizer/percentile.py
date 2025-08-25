from tvm_book.quantization.quantizer.percentile import PercentileQuantizer

def benchmark(
    arr: np.ndarray, 
    nbit: int = 8, 
    percentile: float = 0.99999, 
    iterations: int = 1000
) -> tuple[float, QuantizationResult]:
    """性能测试函数，测量量化器的平均执行时间。
    
    包含预热阶段以减少JIT编译等因素的影响，确保测量结果更准确。
    
    Args:
        arr: 输入数组，将被量化的数据
        nbit: 量化位宽，默认为8
        percentile: 百分位数参数，默认为0.99999
        iterations: 测试迭代次数，默认为1000
        
    Returns:
        一个元组，包含：
            float: 平均执行时间(秒)
            QuantizationResult: 量化结果
    """
    # 创建量化器实例
    quantizer = PercentileQuantizer(nbit=nbit, percentile=percentile)
    
    # 预热
    quantizer.quantize(arr)
    
    # 测试
    start_time = time.time()
    for _ in range(iterations):
        result = quantizer.quantize(arr)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    return avg_time, result
