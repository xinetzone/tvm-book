import numpy as np
import time
from dataclasses import dataclass

@dataclass
class QuantizationResult:
    """存储量化过程中计算得到的缩放因子和零点。
    
    Attributes:
        scale: 缩放因子，用于将量化值映射回原始值域
        zero_point: 零点，用于非对称量化中的偏移校正
    """
    scale: float
    zero_point: int

@dataclass
class QuantizationParams:
    """存储量化过程中需要的配置参数。
    
    Attributes:
        nbit: 量化位宽，决定量化后数据的精度
        percentile: 百分位数参数，用于确定量化范围，范围[0, 1]
    """
    nbit: int
    percentile: float

@dataclass
class PercentileQuantizer:
    """基于百分位数的非对称量化器。
    
    使用百分位数来确定量化范围，适用于处理含有极端值的数据。
    采用非对称量化方式，可以更好地利用量化值域。
    
    Attributes:
        nbit: 量化位宽，默认为8
        percentile: 百分位数参数，用于确定量化范围，范围[0, 1]，
            默认为0.99999
    
    Raises:
        ValueError: 如果percentile不在[0, 1]范围内或nbit非正数
    """
    nbit: int = 8
    percentile: float = 0.99999
    
    def __post_init__(self) -> None:
        """初始化后验证参数的有效性。"""
        # 参数验证
        if not (0 <= self.percentile <= 1):
            raise ValueError("percentile must be between 0 and 1")
        if self.nbit <= 0:
            raise ValueError("nbit must be positive")
    
    @property
    def qmin(self) -> int:
        """计算量化的最小值。
        
        Returns:
            量化后的最小值，基于当前位宽计算
        """
        return -(1 << (self.nbit - 1))
    
    @property
    def qmax(self) -> int:
        """计算量化的最大值。
        
        Returns:
            量化后的最大值，基于当前位宽计算
        """
        return (1 << (self.nbit - 1)) - 1
    
    def quantize(self, arr: np.ndarray) -> QuantizationResult:
        """对输入数组进行量化。
        
        使用百分位数确定量化范围，计算最佳scale和zero_point。
        该方法对含有极端值的数据有较好的鲁棒性。
        
        Args:
            arr: 输入数组，将被量化的数据
            
        Returns:
            QuantizationResult对象，包含计算得到的scale和zero_point
            
        Raises:
            TypeError: 如果arr不是numpy.ndarray
            ValueError: 如果arr为空数组
        """
        # 参数验证
        if not isinstance(arr, np.ndarray):
            raise TypeError("arr must be a numpy.ndarray")
        if arr.size == 0:
            raise ValueError("arr cannot be empty")

        # 简化百分位数计算
        percentile = (1 + self.percentile) / 2
        n = arr.size
        left_idx = int(n * (1 - percentile))
        right_idx = int(n * percentile)
        
        # 边界处理 - 确保索引有效且left_idx <= right_idx
        left_idx = np.clip(left_idx, 0, n-1)
        right_idx = np.clip(right_idx, left_idx, n-1)
        
        # 优化partition操作 - 一次partition获取两个边界值
        # 使用argpartition获取索引，然后直接从原数组取值
        if left_idx == right_idx:
            # 当左右索引相同时，只需要一次partition
            idx = np.argpartition(arr, left_idx)[left_idx]
            left_val = right_val = arr[idx]
        else:
            # 获取左边界
            left_partition_idx = np.argpartition(arr, left_idx)
            left_val = arr[left_partition_idx[left_idx]]
            
            # 从左边界右侧元素中获取右边界
            right_partition_idx = np.argpartition(
                arr[left_partition_idx[left_idx:]], 
                right_idx - left_idx
            )
            right_val = arr[left_partition_idx[left_idx:][
                right_partition_idx[right_idx - left_idx]
            ]]
    
        # 确定min_val和max_val
        min_val = left_val
        max_val = right_val
        min_val = 0 if min_val > 0 else min_val
        max_val = 0 if max_val < 0 else max_val
    
        # 防止除0错误
        if min_val == max_val:
            if min_val == 0:
                return QuantizationResult(scale=1.0, zero_point=0)
            else:
                scale = abs(min_val) / abs(self.qmin)
                return QuantizationResult(scale=scale, zero_point=0)
    
        # 计算scale和zero_point
        scale = (max_val - min_val) / (self.qmax - self.qmin)
        zero_point_float = self.qmin - min_val / scale
        zero_point = int(np.round(zero_point_float))
        zero_point = np.clip(zero_point, self.qmin, self.qmax)
    
        return QuantizationResult(scale=scale, zero_point=zero_point)
