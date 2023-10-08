from utils import set_vta
import tvm

from pathlib import Path
import ctypes


def load_lib():
    """加载库，函数将被注册到 TVM"""
    # 作为全局加载，这样全局 extern symbol 对其他 dll 是可见的。
    curr_path = "./lib/libtvm_ext.so"
    lib = ctypes.CDLL(curr_path, ctypes.RTLD_GLOBAL)
    return lib


_LIB = load_lib()

myadd = tvm.get_global_func("myadd")
print(myadd(4, 5))

@tvm.register_func("tvm.contrib.add")
def add(x, y, z):
    print(f"x:\n{x}\ny:\n{y}")
    tvm.nd.array(x.asnumpy() + y.asnumpy()).copyto(z)

n = 10
A = te.placeholder((10,), name="A")
B = te.placeholder((10,), name="B")
C = te.extern(
    A.shape,
    [A, B],
    lambda ins, outs: tvm.tir.call_packed("tvm.contrib.add", ins[0], ins[1], outs[0]),
    name="C",
)
sch = te.create_schedule(C.op)
te_func = tvm.lower(sch, [A, B, C])
# te_func = te.create_prim_func([A, B])
# te_func.show()
f = tvm.build(te_func, "llvm")
a_np = np.random.uniform(size=(n,)).astype(A.dtype)
b_np = np.random.uniform(size=(n,)).astype(B.dtype)
c_np = a_np + b_np
a = tvm.nd.array(a_np)
b = tvm.nd.array(b_np)
c = tvm.nd.array(np.random.uniform(size=(n,)).astype(C.dtype))
f(a, b, c)
np.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)

# 
x = tvm.ir.make_node("IntImm", dtype="int32", value=10, span=None)
assert isinstance(x, tvm.tir.IntImm)
assert x.value == 10