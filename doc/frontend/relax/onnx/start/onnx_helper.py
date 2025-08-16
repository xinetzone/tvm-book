import numpy as np
import onnx
from onnx import ModelProto, mapping

bg = np.random.MT19937(0)
rg = np.random.Generator(bg)

def generate_random_inputs(
    model: ModelProto, inputs: dict[str, np.ndarray] | None = None
) -> dict[str, np.ndarray]:
    """为ONNX模型生成随机输入数据
    
    参数:
        model: ONNX模型对象(ModelProto 类型)
        inputs: 可选参数，预定义的输入字典，键为输入名称，值为numpy数组
        
    返回:
        包含所有输入名称和对应随机值的字典
    """
    input_values = {}
    # 遍历模型的所有输入节点
    for i in model.graph.input:
        # 如果输入已提供且不为None，则直接使用提供的值
        if inputs is not None and i.name in inputs and inputs[i.name] is not None:
            input_values[i.name] = inputs[i.name]
            continue
            
        # 提取输入张量的形状信息
        shape = [dim.dim_value for dim in i.type.tensor_type.shape.dim]

        # 生成符合形状和数据类型的随机值
        input_values[i.name] = generate_random_value(shape, i.type.tensor_type.elem_type)

    return input_values


def generate_random_value(shape, elem_type) -> np.ndarray:
    """根据形状和数据类型生成随机数值数组
    
    参数:
        shape: 数组形状(tuple/list)
        elem_type: ONNX张量元素类型
        
    返回:
        符合指定形状和数据类型的随机numpy数组
    """
    # 从ONNX类型映射获取对应的numpy数据类型
    if elem_type:
        dtype = str(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[elem_type])
    else:
        dtype = "float32"  # 默认使用float32类型

    # 根据不同类型生成随机值
    if dtype == "bool":
        # 生成布尔型随机值(True/False)
        random_value = rg.choice(a=[False, True], size=shape)
    elif dtype.startswith("int"):
        # 生成整型随机值，并确保非零
        random_value = rg.integers(low=-63, high=63, size=shape).astype(dtype)
        random_value[random_value <= 0] -= 1  # 使所有值非零
    else:
        # 生成浮点型随机值(标准正态分布)
        random_value = rg.standard_normal(size=shape).astype(dtype)

    return random_value
