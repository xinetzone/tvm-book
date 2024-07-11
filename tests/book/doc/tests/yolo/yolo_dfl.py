from tvm.relay.testing import run_infer_type
from tvm.relay.dataflow_pattern import (
    wildcard, is_op, is_tuple,
    is_constant, is_tuple_get_item,
    DFPatternCallback,
    rewrite
)
import tvm
from tvm.relay import transform as _transform
import numpy as np
from tvm.relay.testing import run_infer_type
from tvm.relay.dataflow_pattern import (
    wildcard, is_op,
    is_constant,
    DFPatternCallback,
    rewrite
)
import tvm
from tvm.ir.attrs import DictAttrs
from tvm.relay import transform as _transform
from tvm import relay, te, topi
from tvm.relay.op import op as _op
from tvm.target import generic_func

@generic_func
def schedule_special_op(attrs, outs, target):
    with target:
        # print(f"outs: {outs}")
        # print(f"target: {target}")
        outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
        output = outs[0]
        sch = te.create_schedule(output.op)
        return sch

def custom_yolo_dfl_rel(arg_types, attrs):
    assert len(arg_types) == 1, "type relation arg number mismatch!"
    if attrs:
        assert isinstance(attrs, DictAttrs)
    in_shape = attrs.in_shape
    bbox_size = 4
    assert in_shape[1]%bbox_size == 0
    out_shape = (in_shape[0], bbox_size, in_shape[2])
    return relay.TensorType(out_shape, "float32")

op_name = "vta_special.yolo_dfl"
_op.register(op_name, r"code(cal yolo_dfl.)code")
_op.get(op_name).set_num_inputs(1)
_op.get(op_name).add_argument("data", "Tensor", "The input data tensor.")
_op.get(op_name).set_attrs_type_key("DictAttrs")
_op.get(op_name).add_type_rel(op_name, custom_yolo_dfl_rel)
_op.get(op_name).set_support_level(1)
_op.register_pattern(op_name, _op.OpPattern.COMM_REDUCE)
_op.register_stateful(op_name, False) # 无状态算子

def yolo_dfl(x, channel, in_shape, version="v3", x_scale=-1, x_split=-1):
    attrs = tvm.ir.make_node(
        "DictAttrs",
        channel=channel, in_shape=in_shape, version=version,
        x_scale=x_scale, x_split=x_split,
    )
    return relay.Call(_op.get(op_name), [x], attrs=attrs, type_args=None, span=None)

@_op.register_compute(op_name)
def output_yolo_dfl_compute(attrs, inputs, out_type):
    """yolo_dfl Relay 计算"""
    assert len(inputs) == 1, "输入参数数量不为 1"
    x = inputs[0]
    b, c, a = attrs.in_shape # batch, channels, anchors 
    assert c % 4 == 0
    if x.dtype == "int8":
        x = topi.cast(x, "float32")
        x = topi.multiply(x, attrs.x_scale)
    w = topi.arange(0, attrs.channel, dtype="float32")
    w = topi.reshape(w, (1, attrs.channel, 1, 1))
    if attrs.version == "v3":
        x = topi.reshape(x, (b, c//4, 4, a))
        x = topi.nn.softmax(x, axis=1)
        x = topi.nn.conv2d(x, w, padding=[0, 0, 0, 0], strides=(1, 1), dilation=(1, 1))
        x = topi.reshape(x, (b, 4, a))
    elif attrs.version == "v2":
        x = topi.reshape(x, (b, 4, c//4, a))
        x = topi.transpose(x, [0, 2, 1, 3])
        x = topi.nn.softmax(x, axis=1)
        x = topi.nn.conv2d(x, w, padding=[0, 0, 0, 0], strides=(1, 1), dilation=(1, 1))
        x = topi.reshape(x, (b, 4, a))
    elif attrs.version == "v1":
        x = topi.reshape(x, (b, 4, c//4, a))
        x = topi.transpose(x, [0, 3, 1, 2])
        x = topi.nn.softmax(x, axis=3)
        x = topi.transpose(x, [0, 3, 2, 1])
        x = topi.nn.conv2d(x, w, padding=[0, 0, 0, 0], strides=(1, 1), dilation=(1, 1))
        x = topi.reshape(x, (b, 4, a))
    else:
        raise TypeError(f"暂未支持 {attrs.version}")
    return [x]

_op.register_schedule(op_name, schedule_special_op) # 定义调度

class DFLV1Rewrite(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.reshape = is_op("reshape")(self.x)
        self.transpose = is_op("transpose")(self.reshape).has_attr({"axes": [0, 3, 1, 2]})
        self.softmax = is_op("nn.softmax")(self.transpose).has_attr({"axis": 3})
        self.transpose2 = is_op("transpose")(self.softmax).has_attr({"axes": [0, 3, 2, 1]})
        self.conv_weight = is_constant()
        self.conv = is_op("nn.conv2d")(self.transpose2, self.conv_weight)
        self.reshape2 = is_op("reshape")(self.conv)
        self.pattern = self.reshape2
        

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        conv_weight = node_map[self.conv_weight][0]
        # conv = node_map[self.conv][0]
        b, c, a = _transform.InferTypeLocal(x).shape # batch, channels, anchors 
        conv_weight_shape = _transform.InferTypeLocal(conv_weight).shape
        # print(f"conv_weight_shape[2]: {type(conv_weight_shape[2])}")
        # print(dict(conv.attrs))
        assert conv_weight_shape[0] == conv_weight_shape[2] == conv_weight_shape[3] == 1
        # x = yolo_dfl(x, int(conv_weight_shape[1]), (b, c, a))
        x = yolo_dfl(x, conv_weight_shape[1], (b, c, a), version="v1")
        return x
    
class DFLV2Rewrite(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.reshape = is_op("reshape")(self.x)
        self.transpose = is_op("transpose")(self.reshape).has_attr({"axes": [0, 2, 1, 3]})
        self.softmax = is_op("nn.softmax")(self.transpose).has_attr({"axis": 1})
        self.conv_weight = is_constant()
        self.conv = is_op("nn.conv2d")(self.softmax, self.conv_weight)
        self.reshape2 = is_op("reshape")(self.conv)
        self.pattern = self.reshape2
        

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        conv_weight = node_map[self.conv_weight][0]
        b, c, a = _transform.InferTypeLocal(x).shape # batch, channels, anchors 
        conv_weight_shape = _transform.InferTypeLocal(conv_weight).shape
        assert conv_weight_shape[0] == conv_weight_shape[2] == conv_weight_shape[3] == 1
        x = yolo_dfl(x, conv_weight_shape[1], (b, c, a), version="v2")
        return x
    
    
class DFLV3Rewrite(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.reshape = is_op("reshape")(self.x)
        self.softmax = is_op("nn.softmax")(self.reshape).has_attr({"axis": 1})
        self.conv_weight = is_constant()
        self.conv = is_op("nn.conv2d")(self.softmax, self.conv_weight)
        self.reshape2 = is_op("reshape")(self.conv)
        self.pattern = self.reshape2
        

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        conv_weight = node_map[self.conv_weight][0]
        b, c, a = _transform.InferTypeLocal(x).shape # batch, channels, anchors 
        conv_weight_shape = _transform.InferTypeLocal(conv_weight).shape
        assert conv_weight_shape[0] == conv_weight_shape[2] == conv_weight_shape[3] == 1
        x = yolo_dfl(x, conv_weight_shape[1], (b, c, a), version="v3")
        return x