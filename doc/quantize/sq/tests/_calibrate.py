# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Calibration helpers for Relax quantization."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import tvm
from tvm import relax
from tvm.relax.expr_functor import PyExprMutator
from tvm.relax.testing import vm as rvm
from tvm.runtime import NDArray

TensorLike = Union[np.ndarray, NDArray]


class CalibrationMethod:
    """Supported calibration modes."""

    GLOBAL_SCALE = "global_scale"
    MIN_MAX = "min_max"
    # The following modes are placeholders for future extensions.
    ENTROPY = "entropy"
    KL_DIVERGENCE = "kl_divergence"


@dataclass
class CalibrateContext:
    """Configuration for calibration runs."""

    method: str = CalibrationMethod.GLOBAL_SCALE
    nbit: int = 8
    dtype: str = "int8"
    global_scale: float = 8.0
    percentile: Optional[float] = None
    symmetric: bool = True


def create_calibrate_context(
    method: str = CalibrationMethod.GLOBAL_SCALE,
    *,
    nbit: int = 8,
    dtype: str = "int8",
    global_scale: float = 8.0,
    percentile: Optional[float] = None,
    symmetric: bool = True,
) -> CalibrateContext:
    """Create a calibration context."""

    return CalibrateContext(
        method=method,
        nbit=nbit,
        dtype=dtype,
        global_scale=global_scale,
        percentile=percentile,
        symmetric=symmetric,
    )


def _to_numpy(arr: TensorLike) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    if isinstance(arr, NDArray):
        return arr.numpy() if hasattr(arr, "numpy") else arr.asnumpy()
    raise TypeError(f"Unsupported tensor type: {type(arr)!r}")


def _to_ndarray(value: TensorLike) -> NDArray:
    if isinstance(value, NDArray):
        return value
    if isinstance(value, np.ndarray):
        return tvm.nd.array(value)
    if np.isscalar(value):
        return tvm.nd.array(np.array(value))
    raise TypeError(f"Unsupported value type: {type(value)!r}")


def _dtype_range(dtype: str, nbit: int) -> Tuple[int, int]:
    dtype = dtype.lower()
    if dtype.startswith("int"):
        qmin = -(1 << (nbit - 1))
        qmax = (1 << (nbit - 1)) - 1
    elif dtype.startswith("uint"):
        qmin = 0
        qmax = (1 << nbit) - 1
    else:
        raise ValueError(f"Unsupported dtype for quantization: {dtype}")
    return qmin, qmax


def calculate_scale_zero_point(
    min_val: float,
    max_val: float,
    *,
    dtype: str,
    nbit: int,
    symmetric: bool,
    eps: float = 1e-7,
) -> Tuple[float, int]:
    """Calculate scale and zero point from min/max statistics."""

    qmin, qmax = _dtype_range(dtype, nbit)
    if symmetric:
        max_abs = max(abs(min_val), abs(max_val))
        if max_abs < eps:
            return 1.0, 0 if dtype.startswith("int") else (qmin + qmax) // 2
        scale = max_abs / max(qmax, 1)
        zero = 0 if dtype.startswith("int") else (qmin + qmax) // 2
        return float(scale), int(zero)

    range_val = max(max_val - min_val, eps)
    scale = range_val / max(qmax - qmin, 1)
    zero = round(qmin - min_val / max(scale, eps))
    zero = int(np.clip(zero, qmin, qmax))
    return float(scale), zero


def _default_matcher(op_names: Optional[Sequence[str]]) -> Callable[[relax.Call], bool]:
    names = set(op_names) if op_names else {"nn.conv2d", "nn.dense"}

    def matcher(call: relax.Call) -> bool:
        if not isinstance(call.op, relax.Op):
            return False
        if not isinstance(call.struct_info, relax.TensorStructInfo):
            return False
        return call.op.name in names

    return matcher


class _KeyCounter:
    def __init__(self) -> None:
        self.value = 0

    def next(self) -> int:
        value = self.value
        self.value += 1
        return value


class _ObserverInserter(PyExprMutator):
    """Insert packed function calls that record per-call min/max statistics."""

    def __init__(
        self,
        func: relax.Function,
        matcher: Callable[[relax.Call], bool],
        counter: _KeyCounter,
    ) -> None:
        super().__init__(func)
        self._matcher = matcher
        self._counter = counter

    def visit_call_(self, call: relax.Call) -> relax.Expr:
        call = super().visit_call_(call)
        if not self._matcher(call):
            return call
        key = self._counter.next()
        key_const = relax.const(key, "int32")
        return relax.call_packed("relax.quantize_py.update_minmax", key_const, call)


def _instrument_module(
    mod: tvm.IRModule, matcher: Callable[[relax.Call], bool]
) -> Tuple[tvm.IRModule, int]:
    counter = _KeyCounter()
    new_funcs: Dict[tvm.ir.GlobalVar, relax.BaseFunc] = {}
    for gv, func in sorted(mod.functions_items(), key=lambda item: item[0].name_hint):
        if isinstance(func, relax.Function):
            inserter = _ObserverInserter(func, matcher, counter)
            new_funcs[gv] = inserter.visit_expr(func)
        else:
            new_funcs[gv] = func
    return tvm.IRModule(new_funcs, attrs=mod.attrs), counter.value


_RUNTIME_LOCK = threading.Lock()
_RUNTIME_STATS: Dict[int, Dict[str, float]] = {}
_RUNTIME_REGISTERED = False


def _ensure_runtime_hooks() -> None:
    global _RUNTIME_REGISTERED  # pylint: disable=global-statement
    if _RUNTIME_REGISTERED:
        return

    def _register_once(name: str, func: Callable) -> None:
        if tvm.get_global_func(name, allow_missing=True) is None:
            tvm._ffi.register_func(name, func, override=True)

    def _reset() -> None:
        with _RUNTIME_LOCK:
            _RUNTIME_STATS.clear()

    def _export() -> Dict[int, Dict[str, float]]:
        with _RUNTIME_LOCK:
            return {k: dict(v) for k, v in _RUNTIME_STATS.items()}

    def _update(key: int, arr: NDArray) -> NDArray:
        data = _to_numpy(arr)
        mn = float(np.min(data))
        mx = float(np.max(data))
        with _RUNTIME_LOCK:
            slot = _RUNTIME_STATS.get(key)
            if slot is None:
                _RUNTIME_STATS[key] = {"min": mn, "max": mx}
            else:
                slot["min"] = min(slot["min"], mn)
                slot["max"] = max(slot["max"], mx)
        return arr

    _register_once("relax.quantize_py.reset", _reset)
    _register_once("relax.quantize_py.export", _export)
    _register_once("relax.quantize_py.update_minmax", _update)
    _RUNTIME_REGISTERED = True


def _invoke_vm(
    vm: rvm.VirtualMachine,
    entry: str,
    batch: Union[TensorLike, Sequence[TensorLike], Dict[str, TensorLike]],
    params: Dict[str, TensorLike],
) -> None:
    converted_params = {name: _to_ndarray(val) for name, val in params.items()}
    if isinstance(batch, dict):
        merged = {**converted_params, **{k: _to_ndarray(v) for k, v in batch.items()}}
        vm.set_input(entry, **merged)
        vm.invoke_stateful(entry)
        return

    vm.set_input(entry, **converted_params)
    if isinstance(batch, (tuple, list)):
        inputs = [_to_ndarray(val) for val in batch]
        vm[entry](*inputs)
    else:
        vm[entry](_to_ndarray(batch))


def calibrate_model(
    mod: tvm.IRModule,
    params: Optional[Dict[str, TensorLike]],
    calib_ctx: CalibrateContext,
    dataset: Optional[Iterable],
    *,
    entry: str = "main",
    target: str = "llvm",
    device: Optional[tvm.runtime.Device] = None,
    match_op_names: Optional[Sequence[str]] = None,
) -> Dict[int, Dict[str, Union[float, int, str]]]:
    """Run calibration and return per-call quantization parameters."""

    matcher = _default_matcher(match_op_names)
    device = device or tvm.cpu()

    if calib_ctx.method == CalibrationMethod.GLOBAL_SCALE:
        # No statistics required; quantizer will apply the configured global scale.
        return {}

    if dataset is None:
        raise ValueError("Calibration dataset must be provided when calibration is required.")

    inst_mod, num_observers = _instrument_module(mod, matcher)
    if num_observers == 0:
        return {}

    _ensure_runtime_hooks()
    tvm.get_global_func("relax.quantize_py.reset")()

    executable = relax.build(inst_mod, target=target)
    vm = rvm.VirtualMachine(executable, device)

    params = params or {}
    for batch in dataset:
        _invoke_vm(vm, entry, batch, params)

    stats = tvm.get_global_func("relax.quantize_py.export")()

    qparams: Dict[int, Dict[str, Union[float, int, str]]] = {}
    for key_str, mm in stats.items():
        key = int(key_str)
        scale, zero = calculate_scale_zero_point(
            mm["min"],
            mm["max"],
            dtype=calib_ctx.dtype,
            nbit=calib_ctx.nbit,
            symmetric=calib_ctx.symmetric,
        )
        qparams[key] = {"scale": scale, "zero_point": zero, "dtype": calib_ctx.dtype}
    return qparams


__all__ = [
    "CalibrationMethod",
    "CalibrateContext",
    "create_calibrate_context",
    "calibrate_model",
    "calculate_scale_zero_point",
]
