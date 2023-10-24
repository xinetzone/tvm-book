
# # 定点乘法


import numpy as np
from tvm import te, topi, tir as T, relay
import tvm
from tvm.topi import tag
from tvm.relay.op.op import register_compute, register_shape_func
from tvm.relay.op.op import register_broadcast_schedule, register_injective_schedule
from tvm.relay.op.op import register_pattern, OpPattern
from tvm.topi.utils import get_const_tuple

@tvm.te.tag_scope(tag=tag.ELEMWISE)
def q_multiply_shift(x, y, q, left_shift, right_shift, is_left_shift_required):
    # 当前只支持 int32 类型的数据，并且允许任意数量的通道（lanes）。
    hp_dtype = "int64"
    lp_dtype = "int32"
    assert y.dtype == lp_dtype
    assert left_shift.dtype == lp_dtype
    assert right_shift.dtype == lp_dtype
    one = T.const(1, hp_dtype)
    def _compute(*indices):
        elements = []
        for element in get_const_tuple(axes):
            elements += [indices[element]]
        param_indices = tuple(elements)

        # 0) 获取值
        value = x(*indices)
        multiplier = y(*param_indices)
        ls = left_shift(*param_indices)
        rs = right_shift(*param_indices)

        # 1）将整数乘数进行类型转换并相乘
        value = value.astype(hp_dtype)
        multiplier = multiplier.astype(hp_dtype)
        value = T.Select(T.const(is_left_shift_required, "bool"), 
                         value << ls, value)

        # 2）以更高的精度执行乘法运算
        value = value * multiplier

        # 3)计算舍入标量，用于处理溢出的情况。即实现四舍五入
        total_right_shift = rs + q # 表示在右移运算需要额外增加的位数，以便进行四舍五入。
        pos_rounding_value = (one << (total_right_shift-1)) # 得到只有最高位为 1 的数。
        value = value + pos_rounding_value # 如果 x 的最高位是 1，那么它会被进位；如果 x 的最高位是0，那么它不会被进位。这样就实现了四舍五入的功能。

        # 4）只需将结果向右移位以获得最终输出
        value = value >> total_right_shift
        # 5）定点乘法保持值在 int32 范围内。将其转换回 int32。
        return value.astype(x.dtype)

    return te.compute(x.shape, _compute)


shape = 1, 2
lp_dtype = "int32"
hp_dtype = "int32"
axes = [1]
shift_shape = [shape[ax] for ax in axes]
x = te.placeholder(shape, name="x", dtype=lp_dtype)
y = te.placeholder(shift_shape, name="y", dtype=hp_dtype)
left_shift = te.placeholder(shift_shape, name="left_shift", dtype=hp_dtype)
right_shift = te.placeholder(shift_shape, name="right_shift", dtype=hp_dtype)
# multipliers_shifts = te.placeholder(shape, name="multipliers_shifts", dtype="int32")
z = q_multiply_shift(x, y, 8, left_shift, right_shift, is_left_shift_required=1)
s = te.create_schedule(z.op)
f = tvm.build(s, [x, y, left_shift, right_shift, z], "llvm")
dev = tvm.cpu(0)
a_np = np.ones(shape).astype(x.dtype) * 125
multiplier_np = np.ones(get_const_tuple(y.shape)).astype(hp_dtype) * 36500
ls_np = np.ones(get_const_tuple(left_shift.shape)).astype(hp_dtype) * 8
rs_np = np.ones(get_const_tuple(right_shift.shape)).astype(hp_dtype) * 8
a = tvm.nd.array(a_np, dev) 
multiplier = tvm.nd.array(multiplier_np, dev)
ls = tvm.nd.array(ls_np, dev)
rs = tvm.nd.array(rs_np, dev)
c = tvm.nd.array(np.zeros(get_const_tuple(z.shape), dtype=z.dtype), dev)
f(a, multiplier, ls, rs, c)
print(a, multiplier, ls, rs, c)


x = relay.var("x", shape=(4, 8), dtype="float32")
op0 = relay.qnn.op.quantize(x, relay.const(2.0), relay.const(10), out_dtype="uint8")
op1 = relay.qnn.op.dequantize(op0, relay.const(0.5), relay.const(5))
relay_mod = tvm.IRModule.from_expr(op1)
relay_mod.show()


input_name = "data"
shape = (4, 2)
x = relay.var(input_name, shape=shape, dtype="float32")
op0 = relay.qnn.op.quantize(x, relay.const(np.array([2.0, 2.0])), relay.const(np.array([10, 10])), axis=1, out_dtype="uint8")
op1 = relay.qnn.op.dequantize(op0, relay.const(np.array([0.5, 0.5])), relay.const(np.array([5, 5])))
relay_mod = tvm.IRModule.from_expr(op1)
relay_mod.show()


from tvm_book.tvm_utils.llvm_utils import run_llvm_graph


from tvm.relay.backend import Executor
data_np = np.ones(shape)
inputs = {input_name: data_np}
cpu_outs = run_llvm_graph(relay_mod, {}, inputs)


relay_mod.show()




@tvm.te.tag_scope(tag=tag.ELEMWISE)
def store_multipliers_shifts(x):
    """存储 multipliers_shifts"""
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
