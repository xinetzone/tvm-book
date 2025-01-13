import tvm
from tvm.script import relax as R
from tvm.script import tir as T

@tvm.script.ir_module
class Module:
    @T.prim_func
    def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
        T.func_attr({"global_symbol": "tir_matmul"})
        k = T.int32()
        A = T.match_buffer(x, (32, 32))
        B = T.match_buffer(y, (32, 32))
        C = T.match_buffer(z, (32, 32))

        for i0, j0, k0 in T.grid(32, 32, 32):
            with T.block():
                i, j, k = T.axis.remap("SSR", [i0, j0, k0])
                with T.init():
                    C[i, j] = 0.0
                C[i, j] += A[i, k] * B[j, k]

    @T.prim_func
    def tir_relu(x: T.handle, y: T.handle):
        T.func_attr({"global_symbol": "tir_relu"})
        A = T.match_buffer(x, (32, 32))
        B = T.match_buffer(y, (32, 32))
        for i, j in T.grid(32, 32):
            with T.block():
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = T.max(A[vi, vj], 0.0)

    @T.prim_func
    def tir_zeros(x: T.handle, n: T.int64):
        T.func_attr({"global_symbol": "tir_zeros"})
        A = T.match_buffer(x, [n])
        for i in range(n):
            with T.block():
                vi = T.axis.remap("S", [i])
                A[vi] = 1.0

    @R.function
    def main(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Tuple:
        cls = Module
        with R.dataflow():
            lv0 = R.call_tir(cls.tir_matmul, (x, w), R.Tensor((32, 32), dtype="float32"))
            lv1 = R.call_tir(cls.tir_relu, (lv0), R.Tensor((32, 32), dtype="float32"))
            lv2 = R.call_tir(
                cls.tir_zeros, [], R.Tensor((32,), dtype="float32"), tir_vars=R.ShapeExpr([32])
            )
            gv = (lv1, lv2)
            R.output(gv)
        return gv
