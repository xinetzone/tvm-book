import numpy as np
import time
from tvm_book.quantization.quantizer.percentile import PercentileQuantizer, QuantizationResult

def benchmark(
    arr: np.ndarray, 
    nbit: int = 8, 
    percentile: float = 0.99999, 
    iterations: int = 1000
) -> tuple[float, QuantizationResult]:
    """性能测试函数，测量量化器的平均执行时间。"""
    quantizer = PercentileQuantizer(nbit=nbit, percentile=percentile)
    quantizer.quantize(arr)  # 预热
    
    start_time = time.time()
    for _ in range(iterations):
        result = quantizer.quantize(arr)
    end_time = time.time()
    
    return (end_time - start_time) / iterations, result