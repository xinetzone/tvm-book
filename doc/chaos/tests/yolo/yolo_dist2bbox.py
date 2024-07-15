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

def custom_yolo_dist2bbox_rel(arg_types, attrs):
    assert len(arg_types) == 2, "type relation arg number mismatch!"
    if attrs:
        assert isinstance(attrs, DictAttrs)
    return relay.TensorType(arg_types[0].shape, "float32")

op_name = "vta_special.yolo_dist2bbox"
_op.register(op_name, r"code(cal yolo_dfl.)code")
_op.get(op_name).set_num_inputs(2)
_op.get(op_name).add_argument("data", "Tensor", "The input data tensor.")
_op.get(op_name).add_argument("anchors", "Tensor", "The anchors data tensor.")
_op.get(op_name).set_attrs_type_key("DictAttrs")
_op.get(op_name).add_type_rel(op_name, custom_yolo_dist2bbox_rel)
_op.get(op_name).set_support_level(10)
_op.register_pattern(op_name, _op.OpPattern.COMM_REDUCE)
_op.register_stateful(op_name, False) # 无状态算子

def yolo_dist2bbox(x, anchors, x_scale=-1, x_split=-1):
    attrs = tvm.ir.make_node(
        "DictAttrs",
        x_scale=x_scale, x_split=x_split,
    )
    return relay.Call(_op.get(op_name), [x, anchors], attrs=attrs, type_args=None, span=None)

@_op.register_compute(op_name)
def output_yolo_dist2bbox_compute(attrs, inputs, out_type):
    """yolo_dist2bbox Relay 计算"""
    assert len(inputs) == 2, "输入参数数量不为 2"
    x, anchors = inputs
    x0 = topi.strided_slice(x, begin=[0], end=[2], strides=[1], axes=[1])
    x1 = topi.strided_slice(x, begin=[2], end=[4], strides=[1], axes=[1])
    subtract = topi.subtract(anchors, x0)
    add = topi.add(anchors, x1)
    x = topi.add(subtract, add)
    x = topi.divide(x, 2.0)
    x = topi.concatenate((x, topi.subtract(add, subtract)), axis=1)
    return [x]

_op.register_schedule(op_name, schedule_special_op) # 定义调度

class Dist2BBoxRewrite(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.strided_slice = is_op("strided_slice")(self.x)#.has_attr({"begin": [tvm.runtime.convert(0), tvm.runtime.convert(1)]}) #, "end": [(1<<31) -1, 2]})
        self.anchor_points = is_constant()
        self.subtract = is_op("subtract")(self.anchor_points, self.strided_slice)
        
        self.strided_slice2 = is_op("strided_slice")(self.x)#.has_attr({"begin": [0, 2], "end": [(1<<31) -1, 4]})
        # self.add_const = is_constant()
        self.add = is_op("add")(self.anchor_points, self.strided_slice2)

        self.add2 = is_op("add")(self.subtract, self.add)

        self.divide_const = is_constant()
        self.divide = is_op("divide")(self.add2, self.divide_const)

        self.subtract2 = is_op("subtract")(self.add, self.subtract)

        self.tuple_op = is_tuple((self.divide, self.subtract2))
        self.cat = is_op("concatenate")(self.tuple_op)

        # self.multiply_const = is_constant()
        # self.multiply = is_op("multiply")(self.cat, self.multiply_const)

        self.pattern = self.cat

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        anchor_points = node_map[self.anchor_points][0]
        # add_const = node_map[self.add_const][0]
        divide_const = node_map[self.divide_const][0]
        assert divide_const.data.numpy() == 2.0
        # multiply_const = node_map[self.multiply_const][0]

        # strided_slice = node_map[self.strided_slice][0]
        # strided_slice2 = node_map[self.strided_slice2][0]
        # assert strided_slice.attrs.begin[0] == 0
        # assert strided_slice.attrs.begin[1] == 0
        # assert strided_slice.attrs.end[0] == ((1<<31) - 1)
        # assert strided_slice.attrs.end[1] == 2
        # assert strided_slice2.attrs.begin[0] == 0
        # assert strided_slice2.attrs.begin[1] == 2
        # assert strided_slice2.attrs.end[0] == ((1<<31) - 1)
        # assert strided_slice2.attrs.end[1] == 4
        # 

        # print(f"strided_slice: {dict(strided_slice.attrs)}\n{type(strided_slice.attrs.end[1])}")
        # print(f"add_const=>add_const: {add_const}=>{subtract_const}")
        # a = add_const.data.numpy()
        # b = subtract_const.data.numpy()
        # # a 与 b 均是 anchor_points
        # assert (a!=b).sum() == 0
        # print(multiply_const.data.numpy()[0].tolist())
        _transform.InferTypeLocal(x)
        out = yolo_dist2bbox(x, anchor_points,)
        _transform.InferTypeLocal(out)
        return out
      