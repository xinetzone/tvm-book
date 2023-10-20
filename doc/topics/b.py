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

from tvm.relay.backend import Executor
data_np = np.ones(shape)
inputs = {input_name: data_np}
cpu_outs = run_llvm_graph(relay_mod, {}, inputs)