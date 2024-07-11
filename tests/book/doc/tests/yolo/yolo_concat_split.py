import numpy as np
from tvm.relay.testing import run_infer_type
from tvm.relay.dataflow_pattern import (
    wildcard, is_op,
    is_constant, is_tuple,
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
        print(f"outs: {outs}")
        print(f"target: {target}")
        outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
        output = outs[0]
        sch = te.create_schedule(output.op)
        return sch
    
def custom_yolo_output_concat_split_rel(arg_types, attrs):
    assert len(arg_types) == 1, "type relation arg number mismatch!"
    if attrs:
        assert isinstance(attrs, DictAttrs)
    # print(f"arg_types: {arg_types[0], type(arg_types[0])}")
    assert isinstance(arg_types[0], tvm.ir.type.TupleType)
    x_types = arg_types[0].fields
    x0_shape = x_types[0].shape
    x1_shape = x_types[1].shape
    x2_shape = x_types[2].shape
    x3_shape = x_types[3].shape
    x4_shape = x_types[4].shape
    x5_shape = x_types[5].shape
    assert x0_shape[2] == x1_shape[2] and x0_shape[3] == x1_shape[3]
    assert x2_shape[2] == x3_shape[2] and x2_shape[3] == x3_shape[3]
    assert x4_shape[2] == x5_shape[2] and x4_shape[3] == x5_shape[3]
    assert x0_shape[1]+x1_shape[1] == x2_shape[1]+x3_shape[1] == x4_shape[1]+x5_shape[1]
    b = x0_shape[0]
    c = x0_shape[1]+x1_shape[1]
    anchors = x0_shape[2]*x0_shape[3]+x2_shape[2]*x2_shape[3]+x4_shape[2]*x4_shape[3]
    return relay.TupleType([
        relay.TensorType([b, attrs.y_split, anchors], "float32"),
        relay.TensorType([b, c-attrs.y_split, anchors], "float32")
    ])

op_name = "vta_special.yolo_output_concat_split"
_op.register(op_name, r"code(cal yolo_dfl.)code")
_op.get(op_name).set_num_inputs(1)
_op.get(op_name).add_argument("x", "Tensor", "The inputs data tensor.")
_op.get(op_name).set_attrs_type_key("DictAttrs")
_op.get(op_name).add_type_rel(op_name, custom_yolo_output_concat_split_rel)
_op.get(op_name).set_support_level(10)
_op.register_pattern(op_name, _op.OpPattern.COMM_REDUCE)
_op.register_stateful(op_name, False) # 无状态算子

def yolo_concat_split(x0, x1, x2, x3, x4, x5, 
        x0_scale=-1, x0_split=-1, x1_scale=-1, x1_split=-1, 
        x2_scale=-1, x2_split=-1, x3_scale=-1, x3_split=-1, 
        x4_scale=-1, x4_split=-1, x5_scale=-1, x5_split=-1,
        y_split=-1):
    attrs = tvm.ir.make_node(
        "DictAttrs",
        x0_scale=x0_scale, x0_split=x0_split, x1_scale=x1_scale, x1_split=x1_split, 
        x2_scale=x2_scale, x2_split=x2_split, x3_scale=x3_scale, x3_split=x3_split, 
        x4_scale=x4_scale, x4_split=x4_split, x5_scale=x5_scale, x5_split=x5_split,
        y_split=y_split,
    )
    x = relay.Tuple([x0, x1, x2, x3, x4, x5])
    return relay.Call(_op.get(op_name), [x], attrs=attrs, type_args=None, span=None)

def topi_make_anchors(strides, sizes, dtype, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    for (h, w), stride in zip(sizes, strides):
        sx = topi.arange(0, w, dtype=dtype) + grid_cell_offset  # shift x
        sy = topi.arange(0, h, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = topi.meshgrid((sy, sx), indexing='ij')
        anchor_points.append(topi.reshape(topi.stack((sx, sy), -1), (h*w, 2)))
        stride_tensor.append(topi.full((h * w, 1), fill_value=stride, dtype=dtype))
    return topi.concatenate(anchor_points), topi.concatenate(stride_tensor)

@_op.register_compute(op_name)
def output_yolo_output_concat_split_compute(attrs, inputs, out_type):
    """output_concat_split Relay 计算"""
    assert len(inputs) == 1, "输入参数数量不为 1"
    x0, x1, x2, x3, x4, x5 = inputs[0]
    x01 = topi.concatenate((x0, x1), axis=1)
    c01 = x01.shape[1]
    x01 = topi.reshape(x01, (1, c01, -1))

    x23 = topi.concatenate((x2, x3), axis=1)
    c23 = x23.shape[1]
    x23 = topi.reshape(x23, (1, c23, -1))

    x45 = topi.concatenate((x4, x5), axis=1)
    c45 = x45.shape[1]
    x45 = topi.reshape(x45, (1, c45, -1))

    assert c01 == c23 == c45

    x = topi.concatenate((x01, x23, x45), axis=2)
    x = topi.split(x, attrs.y_split)
    return [x]

_op.register_schedule(op_name, schedule_special_op) # 定义调度

class VTAYoloOutputConcatSplitRewrite(DFPatternCallback):
    """融合concatenate+resahpe+concatenate+split"""
    def __init__(self):
        super().__init__()
        self.x11 = wildcard()
        self.x12 = wildcard()
        self.tuple_op1 = is_tuple((self.x11, self.x12))
        self.cat1 = is_op("concatenate")(self.tuple_op1).has_attr({"axis": 1})
        self.reshape1 = is_op("reshape")(self.cat1)

        self.x21 = wildcard()
        self.x22 = wildcard()
        self.tuple_op2 = is_tuple((self.x21, self.x22))
        self.cat2 = is_op("concatenate")(self.tuple_op2).has_attr({"axis": 1})
        self.reshape2 = is_op("reshape")(self.cat2)

        self.x31 = wildcard()
        self.x32 = wildcard()
        self.tuple_op3 = is_tuple((self.x31, self.x32))
        self.cat3 = is_op("concatenate")(self.tuple_op3).has_attr({"axis": 1})
        self.reshape3 = is_op("reshape")(self.cat3)

        self.tuple_op = is_tuple((self.reshape1, self.reshape2, self.reshape3))
        self.cat = is_op("concatenate")(self.tuple_op).has_attr({"axis": 2})
        self.split = is_op("split")(self.cat)
        self.pattern = self.split

    def callback(self, pre, post, node_map):
        x11 = node_map[self.x11][0]
        x12 = node_map[self.x12][0]

        x21 = node_map[self.x21][0]
        x22 = node_map[self.x22][0]

        x31 = node_map[self.x31][0]
        x32 = node_map[self.x32][0]

        x32 = node_map[self.x32][0]

        tuple_op = node_map[self.tuple_op][0]
        cat = node_map[self.cat][0]
        split = node_map[self.split][0]
        
        x11_shape = _transform.InferTypeLocal(x11).shape
        x12_shape = _transform.InferTypeLocal(x12).shape
        assert x11_shape[2] == x12_shape[2] and x11_shape[3] == x12_shape[3]

        x21_shape = _transform.InferTypeLocal(x21).shape
        x22_shape = _transform.InferTypeLocal(x22).shape
        assert x21_shape[2] == x22_shape[2] and x21_shape[3] == x22_shape[3]

        x31_shape = _transform.InferTypeLocal(x31).shape
        x32_shape = _transform.InferTypeLocal(x32).shape
        assert x31_shape[2] == x32_shape[2] and x31_shape[3] == x32_shape[3]

        _transform.InferTypeLocal(tuple_op)
        _transform.InferTypeLocal(cat)
        split_type = _transform.InferTypeLocal(split)
        out = yolo_concat_split(
            x11, x12, x21, x22, x31, x32, y_split=split_type.fields[0].shape[1]
        )
        _transform.InferTypeLocal(out)
        # out = run_infer_type(out)
        # print(f"out_type: {out_type}")
        return out
