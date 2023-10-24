a_shape = [1, 16, 1]
b_shape = [a_shape[1]]
shape = a_shape
shift_shape = [shape[1]]
x = te.placeholder(shape, name="X", dtype="int32")
y = te.placeholder(shift_shape, name="X", dtype="int32")
l_shift = te.placeholder(shift_shape, name="X", dtype="int32")
r_shift = te.placeholder(shift_shape, name="X", dtype="int32")
out = te.compute(
    shape,
    lambda *index: tvm.tir.q_multiply_shift_per_axis(
        x[index],
        y[index[1]],
        l_shift[index[1]],
        r_shift[index[1]],
        tvm.tir.const(31, "int32"),
        tvm.tir.const(1, "bool"),
        tvm.tir.const(0, "bool"),
    ),
    name="compute",
)
sch = te.create_schedule(out.op)
with tvm.transform.PassContext(opt_level=3):
    host_lib = tvm.build(sch, [x, y, l_shift, r_shift, out], target=tvm.target.Target("llvm"))
# Verify accuracy
x_np = (
    np.random.randint(-1000, 1000, size=np.prod(a_shape)).reshape(a_shape).astype("int32")
)
y_np = (
    np.random.randint(-1000, 1000, size=np.prod(b_shape)).reshape(b_shape).astype("int32")
)
lsh_np = np.random.randint(0, 10, size=np.prod(b_shape)).reshape(b_shape).astype("int32")
rsh_np = np.random.randint(0, 10, size=np.prod(b_shape)).reshape(b_shape).astype("int32")
b_np = (
    np.random.randint(-1000, 1000, size=np.prod(a_shape)).reshape(a_shape).astype("int32")
)
np_args = [x_np, y_np, lsh_np, rsh_np, b_np]
host_args = [tvm.runtime.ndarray.array(arg) for arg in np_args]
host_lib(*host_args)