"""算子 `special_softmax_reshape` 相关实现

示例见：
doc/tutorials/intro/custom-vta-op.ipynb
"""
import tvm
from tvm.ir.attrs import DictAttrs
from tvm import relay, te, topi
from tvm.relay.op import op as _op
from tvm.relay.testing import run_infer_type
from .utils import schedule_special_op

def custom_hard_swish_rel(arg_types, attrs):
    assert len(arg_types) == 1, "type relation arg number mismatch!"
    inputa_type = arg_types[0]
    # if isinstance(inputa_type, tvm.ir.type.IncompleteType):
    #     return tvm.ir.type.IncompleteType()
    return relay.TensorType(inputa_type.shape, dtype="float32")

# 注册算子
op_name = "special.hard_swish"
_op.register(op_name, r"code(cal hard_swish of a tensor.)code")
_op.get(op_name).set_num_inputs(1)
_op.get(op_name).add_argument("data", "Tensor", "The input data tensor.")
_op.get(op_name).set_attrs_type_key("DictAttrs")
# call customized relation functions
_op.get(op_name).add_type_rel(op_name, custom_hard_swish_rel)
_op.get(op_name).set_support_level(1)
_op.register_pattern(op_name, _op.OpPattern.OUT_ELEMWISE_FUSABLE)
_op.register_stateful(op_name, False) # 无状态算子

@_op.register_compute(op_name)
def hard_swish_compute(
        attrs, inputs, out_type
    ):
    """hard_swish TOPI 计算(x*ReLU6(x+3)/6)
    
    Args:
        newshape: 最终输出shape
        axis: softmax 对应的轴
    """
    assert len(inputs) == 1, "输入参数数量不为 1"
    x = inputs[0]
    x_ = topi.max(topi.min(x+3, 6), 0)
    # x_ = topi.clip(x+3, 0, 6)
    x = topi.multiply(x, x_)
    x = topi.multiply(x, 1/6)
    return [x]

def special_hard_swish(x):
    op = _op.get(op_name)
    x = run_infer_type(x)
    return relay.Call(op, [x])

_op.register_schedule(op_name, schedule_special_op) # 定义调度
