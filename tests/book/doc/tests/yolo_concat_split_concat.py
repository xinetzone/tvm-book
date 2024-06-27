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
from yolo_dfl import yolo_dfl
from yolo_dist2bbox import yolo_dist2bbox
from yolo_concat_split import yolo_concat_split

@generic_func
def schedule_special_op(attrs, outs, target):
    with target:
        # print(f"outs: {outs}")
        # print(f"target: {target}")
        outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
        output = outs[0]
        sch = te.create_schedule(output.op)
        return sch

def custom_yolo_yolo_concat_split_concat_rel(arg_types, attrs):
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
    return relay.TensorType([b, c-attrs.y_split+4, anchors], "float32")

op_name = "vta_special.yolo_concat_split_concat"
_op.register(op_name, r"code(cal yolo_concat_split.)code")
_op.get(op_name).set_num_inputs(1)
_op.get(op_name).add_argument("x", "Tensor", "The inputs data tensor.")
_op.get(op_name).set_attrs_type_key("DictAttrs")
_op.get(op_name).add_type_rel(op_name, custom_yolo_yolo_concat_split_concat_rel)
_op.get(op_name).set_support_level(10)
_op.register_pattern(op_name, _op.OpPattern.COMM_REDUCE)
_op.register_stateful(op_name, False) # 无状态算子

def yolo_concat_split_concat(x0, x1, x2, x3, x4, x5, strides, grid_cell_offset=0.5,
        x0_scale=-1, x0_split=-1, x1_scale=-1, x1_split=-1, 
        x2_scale=-1, x2_split=-1, x3_scale=-1, x3_split=-1, 
        x4_scale=-1, x4_split=-1, x5_scale=-1, x5_split=-1,
        y_split=-1):
    attrs = tvm.ir.make_node(
        "DictAttrs", strides=strides, grid_cell_offset=grid_cell_offset,
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
def output_yolo_concat_split_concat_compute(attrs, inputs, out_type):
    """yolo_yolo_concat_split_concat Relay 计算"""
    assert len(inputs) == 1, "输入参数数量不为 1"
    x0, x1, x2, x3, x4, x5 = inputs[0]
    x = yolo_concat_split(x0, x1, x2, x3, x4, x5)
    x0 = yolo_dfl(x[0])
    sizes = (inp.shape[2:4] for inp in [x0, x2, x4])
    anchors, strides = topi_make_anchors(attrs.strides, sizes, x0.dtype, attrs.grid_cell_offset)
    bbox = yolo_dist2bbox(x0, anchors)
    x0 = topi.multiply(bbox, strides)
    x1 = topi.sigmoid(x[1])
    x = topi.concatenate((x0, x1), axis=1)
    return [x]

_op.register_schedule(op_name, schedule_special_op) # 定义调度

class VTAYoloOutputConcatSplitConcatRewrite(DFPatternCallback):
    def __init__(self, require_type=False, rewrite_once=False):
        super().__init__(require_type=require_type, rewrite_once=rewrite_once)
        self.x0 = wildcard()
        self.x1 = wildcard()
        self.x2 = wildcard()
        self.x3 = wildcard()
        self.x4 = wildcard()
        self.x5 = wildcard()
        self.tuple_op0 = is_tuple((self.x0, self.x1, self.x2, self.x3, self.x4, self.x5))
        self.yolo_outputconcat = is_op("vta_special.yolo_output_concat_split")(self.tuple_op0)

        self.tuple_get_item_0 = is_tuple_get_item(self.yolo_outputconcat, 0)
        self.yolo_dfl_call = is_op("vta_special.yolo_dfl")(self.tuple_get_item_0)
        self.anchors = is_constant()
        self.yolo_dist2bbox_call = is_op("vta_special.yolo_dist2bbox")(self.yolo_dfl_call, self.anchors)
        # self.pattern = self.yolo_dist2bbox_call
        
        self.tuple_get_item_1 = is_tuple_get_item(self.yolo_outputconcat, 1)
        self.strides = is_constant()
        self.multiply = is_op("multiply")(self.yolo_dist2bbox_call, self.strides)
        self.sigmoid = is_op("sigmoid")(self.tuple_get_item_1)

        self.tuple_op1 = is_tuple((self.multiply, self.sigmoid))
        self.cat1 = is_op("concatenate")(self.tuple_op1)
        self.pattern = self.cat1

    def callback(self, pre, post, node_map):
        x0 = node_map[self.x0][0]
        x1 = node_map[self.x1][0]
        x2 = node_map[self.x2][0]
        x3 = node_map[self.x3][0]
        x4 = node_map[self.x4][0]
        x5 = node_map[self.x5][0]
        [_transform.InferTypeLocal(x) for x in [x0, x1, x2, x3, x4, x5]]
        anchors = node_map[self.anchors][0].data.numpy()
        strides = node_map[self.strides][0].data.numpy()
        anchors = np.unique(anchors).tolist()
        strides = np.unique(strides).astype("int32").tolist()
        grid_cell_offset = anchors[0]
        # print(f"anchors: {anchors}")
        yolo_outputconcat_call = node_map[self.yolo_outputconcat][0]
        return yolo_concat_split_concat(x0, x1, x2, x3, x4, x5, strides, grid_cell_offset=grid_cell_offset, y_split=yolo_outputconcat_call.attrs.y_split)
      