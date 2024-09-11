"""算子 `special_softmax_reshape` 相关实现

示例见：
doc/tutorials/intro/custom-vta-op.ipynb
"""
import tvm
from tvm.ir.attrs import DictAttrs
from tvm import relay, te, topi
from tvm.relay.op import op as _op
from .utils import schedule_special_op

def custom_hard_sigmoid_rel(arg_types, attrs):
    assert len(arg_types) == 1, "type relation arg number mismatch!"
    inputa_type = arg_types[0]
    return relay.TensorType(inputa_type.shape, dtype="float32")

# 注册算子
op_name = "special.hard_sigmoid"
_op.register(op_name, r"code(cal hard_sigmoid of a tensor.)code")
_op.get(op_name).set_num_inputs(1)
_op.get(op_name).add_argument("data", "Tensor", "The input data tensor.")
_op.get(op_name).set_attrs_type_key("DictAttrs")
# call customized relation functions
_op.get(op_name).add_type_rel(op_name, custom_hard_sigmoid_rel)
_op.get(op_name).set_support_level(1)
_op.register_pattern(op_name, _op.OpPattern.COMM_REDUCE)
_op.register_stateful(op_name, False) # 无状态算子


@_op.register_compute(op_name)
def hard_sigmoid_compute(
        attrs, inputs, out_type
    ):
    """hard_sigmoid TOPI 计算
    
    Args:
        newshape: 最终输出shape
        axis: softmax 对应的轴
    """
    assert len(inputs) == 1, "输入参数数量不为 1"
    x = inputs[0]
    const = 1/6
    x = topi.multiply(x, const)
    x = topi.add(x, 0.5)
    x = topi.clip(x, 0, 1)
    return [x]

def special_hard_sigmoid(x):
    op = _op.get(op_name)
    return relay.Call(op, [x])

_op.register_schedule(op_name, schedule_special_op) # 定义调度
