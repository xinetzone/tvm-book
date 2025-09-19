import torch
import torchvision
from torch import fx
import numpy as np
import tvm
import tvm_ffi
from tvm import relax
from tvm.relax.frontend.torch.fx_translator import from_fx
from tvm.relax.frontend import detach_params
from tvm.relax.testing import vm as rvm

########################################
# 1. 加载 PyTorch ResNet18 并转 FX
########################################
model = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
example_input = torch.randn(1, 3, 224, 224)
fx_mod = fx.symbolic_trace(model)

# 转成 Relax IRModule
input_info = [((1, 3, 224, 224), "float32")]
mod = from_fx(fx_mod, input_info, keep_params_as_input=True)
mod, params = detach_params(mod)

########################################
# 2. Python-only 观察器注册
########################################
import threading
_STATS = {}
_LOCK = threading.Lock()

@tvm_ffi.register_global_func("calib_py.reset")
def _calib_reset():
    global _STATS
    with _LOCK:
        _STATS = {}

@tvm_ffi.register_global_func("calib_py.export_stats")
def _calib_export_stats():
    with _LOCK:
        return {k: {"min": float(v["min"]), "max": float(v["max"])} for k, v in _STATS.items()}

@tvm_ffi.register_global_func("calib_py.update_minmax")
def _calib_update_minmax(key_id: int, arr):
    a = arr.numpy()
    mn, mx = float(np.min(a)), float(np.max(a))
    with _LOCK:
        slot = _STATS.get(key_id)
        if slot is None:
            _STATS[key_id] = {"min": mn, "max": mx}
        else:
            slot["min"] = min(slot["min"], mn)
            slot["max"] = max(slot["max"], mx)
    return arr

########################################
# 3. 插桩 Pass
########################################
class InsertObservers(relax.PyExprMutator):
    def __init__(self, func, match_pred=None, start_key=0):
        super().__init__(func)
        self.match_pred = match_pred
        self.next_key = start_key

    def new_key(self):
        k = self.next_key
        self.next_key += 1
        return k

    def visit_call_(self, call):
        call = super().visit_call_(call)
        if self.match_pred and not self.match_pred(call):
            return call
        if not isinstance(call.struct_info, relax.TensorStructInfo):
            return call
        key_id = self.new_key()
        key_const = relax.const(key_id, "int32")
        return relax.call_packed("calib_py.update_minmax", key_const, call)

def insert_observers_pass(mod, match_pred=None):
    new_funcs = {}
    for gv, func in mod.functions_items():
        if isinstance(func, relax.Function):
            new_funcs[gv] = InsertObservers(func, match_pred).visit_expr(func)
        else:
            new_funcs[gv] = func
    return tvm.IRModule(new_funcs, attrs=mod.attrs)

########################################
# 4. 校准运行
########################################
def run_calibration(mod_obs, params, dataloader, entry="main", target="llvm", dev=tvm.cpu()):
    tvm.get_global_func("calib_py.reset")()
    ex = relax.build(mod_obs, target=target)
    vm = rvm.VirtualMachine(ex, dev)
    if params:
        vm.set_input(entry, **params)
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            vm[entry](*batch)
        else:
            vm[entry](batch)
    return tvm.get_global_func("calib_py.export_stats")()

########################################
# 5. 统计 → 量化参数
########################################
def to_qparams(stats, bits=8, symmetric=True):
    res = {}
    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        for k, mm in stats.items():
            amax = max(abs(mm["min"]), abs(mm["max"]), 1e-6)
            scale = amax / qmax
            res[int(k)] = {"scale": scale, "zero_point": 0, "dtype": "int8"}
    else:
        qmax = (1 << bits) - 1
        for k, mm in stats.items():
            rng = max(mm["max"] - mm["min"], 1e-6)
            scale = rng / qmax
            zp = int(round(-mm["min"] / scale))
            res[int(k)] = {"scale": scale, "zero_point": zp, "dtype": "uint8"}
    return res

########################################
# 6. 替换观察器为 Q/DQ
########################################
def _is_update_minmax(call):
    return (
        isinstance(call.op, relax.ExternFunc)
        and call.op.global_symbol == "call_packed"
        and len(call.args) >= 1
        and isinstance(call.args[0], relax.StringImm)
        and call.args[0].value == "calib_py.update_minmax"
    )

def insert_qdq_from_observers(mod, key2qp, axis=None):
    class Rewriter(relax.PyExprMutator):
        def visit_call_(self, call):
            call = super().visit_call_(call)
            if _is_update_minmax(call):
                key_id = int(call.args[1].value)
                x = call.args[2]
                if key_id not in key2qp:
                    return call
                cfg = key2qp[key_id]
                s_c = relax.const(np.array(cfg["scale"], dtype="float32"))
                zp_c = relax.const(np.array(cfg["zero_point"], dtype="int32"))
                q = relax.op.qdq.quantize(x, s_c, zp_c, axis=axis, out_dtype=cfg["dtype"])
                dq = relax.op.qdq.dequantize(q, s_c, zp_c, axis=axis)
                return dq
            return call
    new_funcs = {}
    for gv, func in mod.functions_items():
        if isinstance(func, relax.Function):
            new_funcs[gv] = Rewriter(func).visit_expr(func)
        else:
            new_funcs[gv] = func
    return tvm.IRModule(new_funcs, attrs=mod.attrs)

########################################
# 7. 端到端执行
########################################
# dataloader: 用随机数据代替
def dataloader(num_batches=4):
    for _ in range(num_batches):
        yield (np.random.randn(1, 3, 224, 224).astype("float32"),)

# 只在 Conv2d 和 Dense 输出插桩
def match_conv_dense(call):
    return isinstance(call.op, relax.Op) and call.op.name in ["nn.conv2d", "nn.dense"]

# 插桩
mod_obs = insert_observers_pass(mod, match_pred=match_conv_dense)
# 校准
stats = run_calibration(mod_obs, params, dataloader())
# 转量化参数
key2qp = to_qparams(stats, bits=8, symmetric=True)
# 替换为 Q/DQ
mod_qdq = insert_qdq_from_observers(mod_obs, key2qp, axis=None)

########################################
# 8. 编译运行量化模型
########################################
ex = relax.build(mod_qdq, target="llvm")
vm = rvm.VirtualMachine(ex, tvm.cpu())
out = vm["main"](np.random.randn(1, 3, 224, 224).astype("float32"))
print("输出 shape:", out.shape)
