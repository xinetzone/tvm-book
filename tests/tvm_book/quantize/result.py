from dataclasses import dataclass

@dataclass
class QuantizationParam:
    """量化参数数据类

    Attributes:
        scale: 缩放因子
        zero_point: 零点
    """
    scale: float
    zero_point: int