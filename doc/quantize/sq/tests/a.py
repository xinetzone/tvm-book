 [markdown]
# # QConfig


from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Any

@dataclass(frozen=True)
class QConfig:
    # 位宽与DType
    nbit_activation: int = 8
    dtype_activation: str = "int8"
    # 校准模式: "global_scale" | "percentile" | "kl_divergence"
    calibrate_mode: str = "global_scale"
    global_scale: float = 8.0
    # 权重尺度策略: "max" | "power2"
    weight_scale: str = "power2"
    # 运行时与图优化
    calibrate_chunk_by: int = -1  # -1 表示按输出层数
    do_simulation: bool = True    # 只做 Q/DQ 假量化

class QCtx:
    Current: Optional[QConfig] = None

from contextlib import contextmanager
@contextmanager
def qconfig(**kwargs):
    old = QCtx.Current
    cfg = QConfig(**({} if old is None else old.__dict__) | kwargs)
    QCtx.Current = cfg
    try:
        yield cfg
    finally:
        QCtx.Current = old

def current_qconfig() -> QConfig:
    assert QCtx.Current is not None, "Use `with qconfig(...):` scope"
    return QCtx.Current



from tvm import relax, transform

def prerequisite_optimize(mod: tvm.IRModule, params: Optional[Dict[str, Any]] = None):
    # 绑定常量参数，便于折叠
    if params:
        main = mod["main"]
        bind_map = {}
        for p in main.params:
            if p.name_hint in params:
                bind_map[p] = relax.const(params[p.name_hint])
        mod = tvm.IRModule({**mod.functions, mod.get_global_var("main"): relax.bind(main, bind_map)})

    seq = transform.Sequential([
        relax.transform.SimplifyExpr(),
        relax.transform.FoldConstant(),
        relax.transform.CanonicalizeBindings(),
        relax.transform.FoldConstant(),
    ])
    with tvm.transform.PassContext(opt_level=3):
        return seq(mod)



import numpy as np
from tvm import relax
from tvm.script import relax as R
from tvm.relax.testing import vm as rvm

def build_profile_module(mod: tvm.IRModule, match_pred):
    # 把每个匹配的张量收集到一个新的 dataflow block 的 tuple 输出
    class Collector(relax.PyExprMutator):
        def __init__(self, func):
            super().__init__(func)
            self.to_collect = []

        def visit_call_(self, call):
            call = super().visit_call_(call)
            if match_pred(call) and isinstance(call.struct_info, relax.TensorStructInfo):
                self.to_collect.append(call)
            return call

        def transform(self):
            # 在函数末尾，把收集到的张量返回为 tuple
            body = self.visit_expr(self.func.body)
            with relax.BlockBuilder() as bb:
                bb.function(self.func.name_hint, self.func.params, None)
                with bb.dataflow():
                    out = bb.emit(relax.Tuple(self.to_collect))
                    bb.output(out)
                bb.emit_func_output(out)
            return bb.get()[bb.get().get_global_var(self.func.name_hint)]

    new_funcs = {}
    for gv, func in mod.functions_items():
        if isinstance(func, relax.Function):
            c = Collector(func)
            new_funcs[gv] = c.transform()
        else:
            new_funcs[gv] = func
    return tvm.IRModule(new_funcs, attrs=mod.attrs)

def collect_stats(mod_profile: tvm.IRModule, dataset: Iterable, entry="main",
                  target="llvm", dev=tvm.cpu(), chunk_by: int = -1):
    # 运行 profile 图：输出为单一 tuple，其中每个 field 对应一层的输出
    ex = relax.build(mod_profile, target=target)
    vm = rvm.VirtualMachine(ex, dev)
    # 先跑一批，确定 tuple 大小
    first = next(iter(dataset))
    if isinstance(first, (tuple, list)):
        out0 = vm[entry](*first)
    elif isinstance(first, dict):
        vm.set_input(entry, **first)
        out0 = vm[entry]()
    else:
        out0 = vm[entry](first)
    num_fields = len(out0)
    # 重新遍历：将所有 batch 的该 tuple 输出累积
    def run_once(batch):
        if isinstance(batch, (tuple, list)):
            return vm[entry](*batch)
        elif isinstance(batch, dict):
            vm.set_input(entry, **batch)
            return vm[entry]()
        else:
            return vm[entry](batch)

    # 分块组织输出，模仿 Relay 的 chunking
    chunk = num_fields if chunk_by == -1 else chunk_by
    results = []
    for s in range(0, num_fields, chunk):
        e = min(s + chunk, num_fields)
        per_field = [[] for _ in range(e - s)]
        # 再跑一遍所有数据
        for batch in dataset:
            outs = run_once(batch)
            # 提取分块内字段，转为 1D
            for i, idx in enumerate(range(s, e)):
                per_field[i].append(outs[idx].numpy().reshape(-1))
        # 拼接每层的样本
        results.append([np.concatenate(per_field[i]) for i in range(e - s)])
    return results  # List[List[np.ndarray]]: blocks x fields_in_block



import multiprocessing as mp

def _find_scale_by_percentile(arr, percentile=0.99999):
    x = np.abs(arr)
    k = int(x.size * percentile)
    return np.partition(x, k)[k]

# KL 请自行实现/移植（Relay 用了 kl_divergence._find_scale_by_kl）
def _find_scale_by_kl(arr):
    raise NotImplementedError("KL implementation omitted for brevity")

def calibrate_scales(mod_profile, dataset, mode="percentile", chunk_by=-1):
    blocks = collect_stats(mod_profile, dataset, chunk_by=chunk_by)
    scales = []
    for samples in blocks:
        if mode == "global_scale":
            # 每层同一常数（更常见做法：全局一次返回）
            gs = current_qconfig().global_scale
            scales += [gs for _ in samples]
        elif mode == "percentile":
            with mp.Pool() as pool:
                scales += list(pool.map(_find_scale_by_percentile, samples))
        elif mode == "kl_divergence":
            with mp.Pool() as pool:
                scales += list(pool.map(_find_scale_by_kl, samples))
        else:
            raise ValueError(f"Unknown calibrate mode: {mode}")
    return scales  # 顺序与 profile tuple 字段一致



def weight_scale_max(w_np: np.ndarray):
    return float(np.max(np.abs(w_np))) if w_np.size > 0 else 1.0

def weight_scale_power2(w_np: np.ndarray):
    val = weight_scale_max(w_np)
    if val <= 0:
        return 1.0
    return float(2 ** np.ceil(np.log2(val)))



from tvm import relax

def insert_qdq_by_profile_order(mod: tvm.IRModule, scales, bits=8, dtype="int8", axis=None):
    qmax = (1 << (bits - 1)) - 1
    idx_ref = {"i": 0}

    class Rewriter(relax.PyExprMutator):
        def __init__(self, func):
            super().__init__(func)
            self.collected = []

        def visit_call_(self, call: relax.Call):
            call = super().visit_call_(call)
            # 用与 profile 同样的 match 规则；这里简化：对所有张量 call 插入
            if isinstance(call.struct_info, relax.TensorStructInfo):
                i = idx_ref["i"]; idx_ref["i"] += 1
                if i < len(scales):
                    scale = max(scales[i], 1e-6) / qmax
                    s_c = relax.const(np.array(scale, dtype="float32"))
                    zp_c = relax.const(np.array(0, dtype="int32"))
                    q = relax.op.qdq.quantize(call, s_c, zp_c, axis=axis, out_dtype=dtype)
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



def quantize_relax_with_calibration(
    mod: tvm.IRModule,
    params=None,
    dataset=None,
    match_pred=lambda call: isinstance(call.struct_info, relax.TensorStructInfo),
    target="llvm",
    dev=tvm.cpu(),
):
    # 1) 前置优化
    mod = prerequisite_optimize(mod, params)

    # 2) 构建 profile 图（把观察到的张量聚合成 tuple 输出）
    mod_profile = build_profile_module(mod, match_pred)

    # 3) 收集样本并求尺度
    cfg = current_qconfig()
    scales = calibrate_scales(
        mod_profile,
        dataset,
        mode=cfg.calibrate_mode,
        chunk_by=cfg.calibrate_chunk_by,
    )

    # 4) 插入 Q/DQ（逐张量对称）
    mod_qdq = insert_qdq_by_profile_order(
        mod, scales,
        bits=cfg.nbit_activation,
        dtype=cfg.dtype_activation,
        axis=None
    )
    return mod_qdq



with qconfig(
    nbit_activation=8,
    dtype_activation="int8",
    calibrate_mode="percentile",  # "global_scale" | "kl_divergence"
    global_scale=8.0,
    weight_scale="power2",
    calibrate_chunk_by=64,
    do_simulation=True,
):
    mod_qdq = quantize_relax_with_calibration(mod, params=..., dataset=..., target="llvm", dev=tvm.cpu())



import torch
import torchvision

# 加载预训练 ResNet18
model = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()

# 随机输入（假设输入尺寸 224x224）
example_input = torch.randn(1, 3, 224, 224)

# 导出 TorchScript
scripted_model = torch.jit.trace(model, example_input).eval()



import tvm
from tvm import relax
from tvm.relax.frontend import from_pytorch

input_shape = (1, 3, 224, 224)
input_info = [("input0", input_shape)]
mod, params = from_pytorch(scripted_model, input_info)






