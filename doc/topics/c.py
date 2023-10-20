
# # 自定义 TOPI 算子


import numpy as np
from tvm import te, topi, tir as T, relay
import tvm
from tvm.topi import tag
from tvm.relay import op
from tvm.relay.op.op import register_compute, register_shape_func
from tvm.relay.op.op import register_broadcast_schedule, register_injective_schedule
from tvm.relay.op.op import register_pattern, OpPattern
from tvm.topi.utils import get_const_tuple
from tvm.relay.testing import run_opt_pass
import vta
from tvm_book.tvm_utils.llvm_utils import run_llvm_graph
from vta.top.graphpack import (
    _channel_const_match,
    _get_tensor_type,
)

def _channel_shape_match(data, dshape, cfactor):
    """pad 0 以对齐维度 """
    dshape =  list(dshape)
    pad_width_diff, dshape[1] = _channel_const_match(dshape[1], cfactor)
    if pad_width_diff != 0:
        pad_width = len(dshape) * [[0, 0]]
        pad_width[1] = [0, pad_width_diff]
        data = op.nn.pad(data, pad_width)
        data = run_opt_pass(data, relay.transform.InferType())
    return data, dshape


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def vta_preprocessing(x):
    """数据预处理"""
    # hp_dtype = "int64"
    # lp_dtype = "int32"
    # assert y.dtype == lp_dtype
    # assert left_shift.dtype == lp_dtype
    # assert right_shift.dtype == lp_dtype
    # one = T.const(1, hp_dtype)
    def _compute(*indices):
        # elements = []
        # for element in get_const_tuple(axes):
        #     elements += [indices[element]]
        # param_indices = tuple(elements)

        # 0) 获取值
        value = x(*indices)
        
        return value.astype(x.dtype)

    return te.compute(x.shape, _compute)


shape = 1, 16, 224, 224
dtype = "float32"
x = te.placeholder(shape, name="data", dtype=dtype)
y = vta_preprocessing(x)
s = te.create_schedule(y.op)
mod = tvm.lower(s, [x, y])
mod = relay.transform.InferType()(mod)
mod.show()





cfactor = 16
a_min, a_max = -127, 127
dtype = "float32"
out_dtype = "int8"
shape = 1, 3, 224, 224
pad_channel = shape[1] - cfactor
pad_width = [(0, 0), (0, pad_channel), (0, 0), (0, 0)]
x = relay.var("data", shape=shape, dtype=dtype)
const = relay.var("scale", shape=(), dtype=dtype)
y = x * const
y = op.round(y)
y = op.clip(y, a_min, a_max)
y = op.cast(y, out_dtype)
y, shape = _channel_shape_match(y, shape, cfactor)
# new_fn = relay.Function([x], y)
mod = tvm.IRModule.from_expr(y)
mod = relay.transform.InferType()(mod)
mod.show()





print(new_fn)




tvm.IRModule.from_expr(func).show()
intrp = relay.create_executor("graph", device=tvm.cpu(0), target="llvm")

data_np = np.arange(np.prod(shape)).reshape(shape).astype("float32")
op_res, new_data = intrp.evaluate(func)(data_np)
np.testing.assert_allclose(data_np, new_data.numpy())


def before():
    x = relay.var("x", shape=(1, 64, 56, 56))
    channels = 128
    weight = relay.var("weight", shape=(channels, 64, 1, 1))
    y = relay.nn.conv2d(
        x,
        weight,
        channels=channels,
        kernel_size=(1, 1),
        padding=(1, 1),
        data_layout="NCWH",
        kernel_layout="OIHW",
    )
    y = relay.nn.relu(y)
    y = relay.Function([x, weight], y)
    return y

a = before()
a = run_opt_pass(a, relay.transform.ConvertLayout({"nn.conv2d": ["NCHW1n16c", "OIHW16o16i"]}))
print(a)











