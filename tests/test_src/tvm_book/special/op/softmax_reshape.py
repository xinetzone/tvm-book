"""算子 `special_softmax_reshape` 相关实现

示例见：
doc/tutorials/intro/custom-op.ipynb
"""
import tvm
from tvm.ir.attrs import DictAttrs
from tvm import relay, te, topi
from tvm.relay.op import op as _op
from .utils import schedule_special_op

    
def topi_softmax_reshape(x: te.Tensor, newshape: list, axis: int=1):
    """softmax_reshape TOPI 计算
    
    Args:
        newshape: 最终输出shape
        axis: softmax 对应的轴
    """
    x = topi.nn.softmax(x, axis=axis)
    x = topi.reshape(x, newshape=newshape)
    return x

def custom_softmax_reshape_rel(arg_types, attrs):
    assert len(arg_types) == 1, "type relation arg number mismatch!"
    if attrs:
        assert isinstance(attrs, DictAttrs)
    inputa_type = arg_types[0]
    shape = inputa_type.shape
    shape = shape[0], shape[1] * shape[2] * shape[3]
    return relay.TensorType(shape, inputa_type.dtype)

# 注册算子
op_name = "special.softmax_reshape"
_op.register(op_name, r"code(cal softmax_reshape of a tensor.)code")
_op.get(op_name).set_num_inputs(1)
_op.get(op_name).add_argument("data", "Tensor", "The input data tensor.")
_op.get(op_name).set_attrs_type_key("DictAttrs")
# call customized relation functions
_op.get(op_name).add_type_rel(op_name, custom_softmax_reshape_rel)
_op.get(op_name).set_support_level(1)
_op.register_pattern(op_name, _op.OpPattern.COMM_REDUCE)
_op.register_stateful(op_name, False) # 无状态算子

@_op.register_compute(op_name)
def output_softmax_reshape_compute(attrs, inputs, out_type):
    """softmax_reshape Relay 计算"""
    assert len(inputs) == 1, "输入参数数量不为 1"
    x = inputs[0]
    x = topi_softmax_reshape(x, axis=int(attrs.axis), newshape=attrs.get_int_tuple("newshape"))
    return [x]

def special_softmax_reshape(x, axis, newshape):
    op = _op.get(op_name)
    attrs = tvm.ir.make_node(
        "DictAttrs", 
        axis=axis,
        newshape=newshape,
    )
    return relay.Call(op, [x], attrs=attrs)

_op.register_schedule(op_name, schedule_special_op) # 定义调度
