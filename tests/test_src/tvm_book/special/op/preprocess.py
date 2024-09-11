"""算子 `special_softmax_reshape` 相关实现

示例见：
doc/tutorials/intro/custom-vta-op.ipynb
"""
import tvm
from tvm.ir.attrs import DictAttrs
from tvm import relay, te, topi
from tvm.relay.op import op as _op
from .utils import schedule_special_op

def custom_preprocess_rel(arg_types, attrs):
    assert len(arg_types) == 2, "type relation arg number mismatch!"
    if attrs:
        assert isinstance(attrs, DictAttrs)
        dtype = attrs.dtype
    else:
        dtype = inputa_type.dtype
    inputa_type = arg_types[0]
    return relay.TensorType(inputa_type.shape, dtype)

# 注册算子
op_name = "special.preprocess"
_op.register(op_name, r"code(cal preprocess of a tensor.)code")
_op.get(op_name).set_num_inputs(2)
_op.get(op_name).add_argument("data", "Tensor", "The input data tensor.")
_op.get(op_name).add_argument("scale", "Tensor", "The input data scale tensor.")
_op.get(op_name).set_attr("dtype", "int8")
_op.get(op_name).set_attrs_type_key("DictAttrs")
# call customized relation functions
_op.get(op_name).add_type_rel(op_name, custom_preprocess_rel)
_op.get(op_name).set_support_level(1)
_op.register_pattern(op_name, _op.OpPattern.COMM_REDUCE)
_op.register_stateful(op_name, False) # 无状态算子

@_op.register_compute(op_name)
def preprocess_compute(
        attrs, inputs, out_type
    ):
    """softmax_reshape TOPI 计算
    
    Args:
        newshape: 最终输出shape
        axis: softmax 对应的轴
    """
    assert len(inputs) == 2, "输入参数数量不为 1"
    x, scale = inputs
    x = topi.multiply(x, scale)
    x = topi.round(x)
    x = topi.clip(x, attrs.a_min, attrs.a_max)
    x = topi.cast(x, dtype=out_type.dtype)
    return [x]

def special_preprocess(x, scale, a_min, a_max, dtype):
    op = _op.get(op_name)
    attrs = tvm.ir.make_node(
        "DictAttrs",
        a_min=a_min, 
        a_max=a_max,
        dtype=dtype,
    )
    return relay.Call(op, [x, scale], attrs=attrs)

_op.register_schedule(op_name, schedule_special_op) # 定义调度
