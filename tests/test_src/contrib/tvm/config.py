from dataclasses import dataclass


@dataclass
class Config:
    model: str  # 模型名称
    expected_acc: float
    batch_size: int = 1


@dataclass
class QConfig:
    model: str  # 模型名称
    expected_acc: float
    nbit_input: int = 8
    dtype_input: str = "int8"
    nbit_output: int = 16
    dtype_output: str = "int16"
    global_scale: float = 4.0
    batch_size: int = 1
