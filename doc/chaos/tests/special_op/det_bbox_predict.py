import tvm
from tvm.ir.attrs import DictAttrs
from tvm.relay import transform as _transform
from tvm import relay, te, topi
from tvm.relay.op import op as _op
from .utils import schedule_special_op

def custom_det_bbox_predict_rel(arg_types, attrs):
    assert len(arg_types) == 3, "type relation arg number mismatch!"
    if attrs:
        assert isinstance(attrs, DictAttrs)

    bbox_size = 4
    n0, c0, h0, w0 = arg_types[0].shape
    n1, c1, h1, w1 = arg_types[1].shape
    n2, c2, h2, w2 = arg_types[2].shape
    assert n0 == n1 == n2
    assert c0 % bbox_size == 0
    assert c1 % bbox_size == 0
    assert c2 % bbox_size == 0
    return relay.TensorType((n0, (c0*h0*w0+c1*h1*w1+c2*h2*w2)//bbox_size, bbox_size), "float32")

op_name = "llvm_special.det_bbox_predict"
_op.register(op_name, r"code(cal yolo_concat_split.)code")
_op.get(op_name).set_num_inputs(3)
_op.get(op_name).add_argument("x0", "Tensor", "The inputs data tensor.")
_op.get(op_name).add_argument("x1", "Tensor", "The inputs data tensor.")
_op.get(op_name).add_argument("x2", "Tensor", "The inputs data tensor.")
_op.get(op_name).set_attrs_type_key("DictAttrs")
_op.get(op_name).add_type_rel(op_name, custom_det_bbox_predict_rel)
_op.get(op_name).set_support_level(10)
_op.register_pattern(op_name, _op.OpPattern.COMM_REDUCE)
_op.register_stateful(op_name, False) # 无状态算子

def det_bbox_predict(x0, x1, x2):
    return relay.Call(_op.get(op_name), [x0, x1, x2], attrs=None, type_args=None, span=None)

@_op.register_compute(op_name)
def det_bbox_predict_compute(attrs, inputs, out_type):
    """det_bbox_predict Relay 计算"""
    assert len(inputs) == 3, "输入参数数量不为 3"
    x0, x1, x2 = inputs
    bbox_size = 4
    x0 = topi.transpose(x0, [0, 2, 3, 1])
    x0 = topi.reshape(x0, [1, x0.shape[1]*x0.shape[2]*x0.shape[3]//bbox_size, bbox_size])
    x1 = topi.transpose(x1, [0, 2, 3, 1])
    x1 = topi.reshape(x1, [1, x1.shape[1]*x1.shape[2]*x1.shape[3]//bbox_size, bbox_size])
    x2 = topi.transpose(x2, [0, 2, 3, 1])
    x2 = topi.reshape(x2, [1, x2.shape[1]*x2.shape[2]*x2.shape[3]//bbox_size, bbox_size])
    x = topi.concatenate([x0, x1, x2], axis=1)
    return [x]

_op.register_schedule(op_name, schedule_special_op) # 定义调度
