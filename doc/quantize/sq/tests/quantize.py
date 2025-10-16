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
"""Relax quantization utilities inspired by Relay v0.19.0 quantize flow."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import tvm
from tvm import relax
from tvm.ir import IRModule
from tvm.relax import expr as relax_expr
from tvm.relax.expr_functor import PyExprMutator
from tvm.relax.op import qdq
from tvm.relax.transform import ModulePass
from tvm.runtime import NDArray

from ._calibrate import (
    CalibrateContext,
    CalibrationMethod,
    calculate_scale_zero_point,
    calibrate_model,
    create_calibrate_context,
)

TensorLike = Union[np.ndarray, NDArray]


@dataclass(frozen=True)
class QuantizeConfig:
    """Configuration knobs for Relax quantization."""

    calibrate_mode: str = CalibrationMethod.GLOBAL_SCALE
    global_scale: float = 8.0
    nbit_input: int = 8
    nbit_weight: int = 8
    dtype_input: str = "int8"
    dtype_weight: str = "int8"
    dtype_activation: str = "int8"
    quantize_type: str = "int"  # "int" for symmetric, "uint" for asymmetric
    weight_axis: Optional[int] = 0
    activation_axis: Optional[int] = None
    per_channel_weights: bool = True
    per_channel_activations: bool = False
    quantize_inputs: bool = False
    quantize_weights: bool = True
    quantize_outputs: bool = True
    skip_dense_layer: bool = False
    skip_conv_layers: bool = False
    match_op_names: Optional[Sequence[str]] = None
    percentile: Optional[float] = None
    symmetric_weights: bool = True
    symmetric_activations: bool = True


def create_quantize_config(**kwargs) -> QuantizeConfig:
    """Create a :class:`QuantizeConfig` with overrides."""

    config = QuantizeConfig(**kwargs)
    symmetric = config.quantize_type == "int"
    return replace(
        config,
        symmetric_weights=config.symmetric_weights if "symmetric_weights" in kwargs else symmetric,
        symmetric_activations=(
            config.symmetric_activations if "symmetric_activations" in kwargs else symmetric
        ),
    )


class QuantizeContext:
    """Thread-local context that stores the active quantization configuration."""

    _current: Optional["QuantizeContext"] = None

    def __init__(self, config: QuantizeConfig) -> None:
        self.config = config
        self._old: Optional["QuantizeContext"] = None

    def __enter__(self) -> "QuantizeContext":
        self._old = QuantizeContext._current
        QuantizeContext._current = self
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[override]
        QuantizeContext._current = self._old

    @staticmethod
    def current() -> Optional["QuantizeContext"]:
        return QuantizeContext._current


def qconfig(
    config: Optional[QuantizeConfig] = None,
    **kwargs,
) -> QuantizeContext:
    """Context manager that installs a quantization configuration."""

    if config is None:
        config = create_quantize_config(**kwargs)
    elif kwargs:
        config = replace(config, **kwargs)
    return QuantizeContext(config)


def _to_numpy(value: TensorLike) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, NDArray):
        return value.numpy() if hasattr(value, "numpy") else value.asnumpy()
    raise TypeError(f"Unsupported tensor type: {type(value)!r}")


def _resolve_param_ndarray(params: Dict[str, TensorLike], name: str) -> Optional[np.ndarray]:
    if name not in params:
        return None
    return _to_numpy(params[name])


def _weight_arg_index(op_name: str) -> Optional[int]:
    if op_name in {"nn.conv2d", "nn.dense", "nn.matmul"}:
        return 1
    return None


def _default_matcher(op_names: Optional[Sequence[str]]):
    names = set(op_names) if op_names else {"nn.conv2d", "nn.dense"}

    def matcher(call: relax.Call) -> bool:
        if not isinstance(call.op, relax.Op):
            return False
        if call.op.name not in names:
            return False
        return isinstance(call.struct_info, relax.TensorStructInfo)

    return matcher


def _compute_weight_qparams(
    weight: np.ndarray,
    *,
    axis: Optional[int],
    dtype: str,
    nbit: int,
    symmetric: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if axis is None:
        min_val = float(weight.min()) if weight.size else 0.0
        max_val = float(weight.max()) if weight.size else 0.0
        scale, zero = calculate_scale_zero_point(
            min_val,
            max_val,
            dtype=dtype,
            nbit=nbit,
            symmetric=symmetric,
        )
        return np.array(scale, dtype="float32"), np.array(zero, dtype="int32")

    axis_norm = axis if axis >= 0 else weight.ndim + axis
    reduce_axes = tuple(i for i in range(weight.ndim) if i != axis_norm)
    if not reduce_axes:
        min_vals = weight.copy()
        max_vals = weight.copy()
    else:
        min_vals = weight.min(axis=reduce_axes)
        max_vals = weight.max(axis=reduce_axes)
    scales = np.empty_like(min_vals, dtype="float32")
    zeros = np.empty_like(min_vals, dtype="int32")
    it = np.nditer(min_vals, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        scale, zero = calculate_scale_zero_point(
            float(min_vals[idx]),
            float(max_vals[idx]),
            dtype=dtype,
            nbit=nbit,
            symmetric=symmetric,
        )
        scales[idx] = scale
        zeros[idx] = zero
        it.iternext()
    return scales.astype("float32"), zeros.astype("int32")


def _default_activation_qparams(config: QuantizeConfig) -> Tuple[float, int]:
    qmin, qmax = (_dtype_range(config.dtype_activation, config.nbit_input))
    if config.symmetric_activations:
        if qmax == 0:
            return 1.0, 0
        scale = float(config.global_scale) / max(qmax, 1)
        zero = 0 if config.dtype_activation.startswith("int") else (qmin + qmax) // 2
        return scale, zero

    min_val = -config.global_scale
    max_val = config.global_scale
    scale, zero = calculate_scale_zero_point(
        min_val,
        max_val,
        dtype=config.dtype_activation,
        nbit=config.nbit_input,
        symmetric=False,
    )
    return scale, zero


def _dtype_range(dtype: str, nbit: int) -> Tuple[int, int]:
    dtype = dtype.lower()
    if dtype.startswith("int"):
        return -(1 << (nbit - 1)), (1 << (nbit - 1)) - 1
    if dtype.startswith("uint"):
        return 0, (1 << nbit) - 1
    raise ValueError(f"Unsupported dtype: {dtype}")


class Quantizer(PyExprMutator):
    """Mutator that inserts QDQ pairs around Relax ops."""

    def __init__(
        self,
        mod: IRModule,
        params: Dict[str, TensorLike],
        config: QuantizeConfig,
        call_qparams: Dict[int, Dict[str, Union[float, int, str]]],
        matcher: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.mod = mod
        self.params = params
        self.config = config
        self.call_qparams = call_qparams
        self._matcher = _default_matcher(matcher or config.match_op_names)
        self._call_counter = 0

    def transform(self) -> IRModule:
        new_funcs: Dict[tvm.ir.GlobalVar, relax.BaseFunc] = {}
        self._call_counter = 0
        for gv, func in sorted(self.mod.functions_items(), key=lambda item: item[0].name_hint):
            if isinstance(func, relax.Function):
                new_funcs[gv] = self.visit_expr(func)
            else:
                new_funcs[gv] = func
        return tvm.IRModule(new_funcs, attrs=self.mod.attrs)

    def visit_call_(self, call: relax.Call) -> relax.Expr:
        call = super().visit_call_(call)
        if not self._matcher(call):
            return call

        op_name = call.op.name if isinstance(call.op, relax.Op) else ""
        args = list(call.args)
        weight_idx = _weight_arg_index(op_name)

        if (
            self.config.quantize_weights
            and weight_idx is not None
            and not self._should_skip_op(op_name)
        ):
            args[weight_idx] = self._quantize_weight(args[weight_idx])

        new_call = relax.Call(call.op, args, call.attrs, call.type_args, call.span)
        result = new_call

        if self.config.quantize_outputs and not self._should_skip_op(op_name):
            call_id = self._call_counter
            self._call_counter += 1
            result = self._quantize_activation(result, call_id)
        else:
            self._call_counter += 1

        return result

    def _should_skip_op(self, op_name: str) -> bool:
        if op_name == "nn.conv2d" and self.config.skip_conv_layers:
            return True
        if op_name == "nn.dense" and self.config.skip_dense_layer:
            return True
        return False

    def _quantize_weight(self, expr: relax.Expr) -> relax.Expr:
        axis = self.config.weight_axis if self.config.per_channel_weights else None
        weight_np: Optional[np.ndarray] = None

        if isinstance(expr, relax_expr.Var):
            weight_np = _resolve_param_ndarray(self.params, expr.name_hint)
        elif isinstance(expr, relax_expr.Constant):
            weight_np = _to_numpy(expr.data)

        if weight_np is None:
            # Without static weights, fall back to runtime quantization.
            return expr

        scales, zeros = _compute_weight_qparams(
            weight_np,
            axis=axis,
            dtype=self.config.dtype_weight,
            nbit=self.config.nbit_weight,
            symmetric=self.config.symmetric_weights,
        )

        scale_const = relax.const(scales.astype("float32"))
        zero_const = relax.const(zeros.astype("int32"))

        quantized = qdq.quantize(
            expr,
            scale_const,
            zero_const,
            axis=axis,
            out_dtype=self.config.dtype_weight,
        )
        return qdq.dequantize(
            quantized,
            scale_const,
            zero_const,
            axis=axis,
        )

    def _quantize_activation(self, expr: relax.Expr, call_id: int) -> relax.Expr:
        axis = self.config.activation_axis if self.config.per_channel_activations else None
        qparam = self.call_qparams.get(call_id)

        if qparam:
            scale = qparam["scale"]
            zero = qparam["zero_point"]
            out_dtype = qparam.get("dtype", self.config.dtype_activation)
        else:
            scale, zero = _default_activation_qparams(self.config)
            out_dtype = self.config.dtype_activation

        scale_arr = np.array(scale, dtype="float32")
        zero_arr = np.array(zero, dtype="int32")

        scale_const = relax.const(scale_arr)
        zero_const = relax.const(zero_arr)

        q = qdq.quantize(expr, scale_const, zero_const, axis=axis, out_dtype=out_dtype)
        return qdq.dequantize(q, scale_const, zero_const, axis=axis)


def quantize(
    mod: IRModule,
    params: Optional[Dict[str, TensorLike]] = None,
    *,
    dataset: Optional[Iterable] = None,
    config: Optional[QuantizeConfig] = None,
    entry: str = "main",
    target: str = "llvm",
    device: Optional[tvm.runtime.Device] = None,
    match_op_names: Optional[Sequence[str]] = None,
) -> IRModule:
    """Quantize a Relax module by inserting QDQ pairs."""

    ctx = QuantizeContext.current()
    active_config = config or (ctx.config if ctx is not None else QuantizeConfig())
    params = params or {}

    calibrate_ctx = create_calibrate_context(
        active_config.calibrate_mode,
        nbit=active_config.nbit_input,
        dtype=active_config.dtype_activation,
        global_scale=active_config.global_scale,
        percentile=active_config.percentile,
        symmetric=active_config.quantize_type == "int",
    )

    call_qparams: Dict[int, Dict[str, Union[float, int, str]]] = {}
    if active_config.calibrate_mode != CalibrationMethod.GLOBAL_SCALE:
        call_qparams = calibrate_model(
            mod,
            params,
            calibrate_ctx,
            dataset,
            entry=entry,
            target=target,
            device=device,
            match_op_names=match_op_names or active_config.match_op_names,
        )

    quantizer = Quantizer(
        mod,
        params,
        active_config,
        call_qparams,
        matcher=match_op_names or active_config.match_op_names,
    )
    return quantizer.transform()


@tvm.transform.register_transform_pass(opt_level=0)
class QuantizePass(ModulePass):
    """Module pass wrapper for Relax quantization."""

    def __init__(
        self,
        config: QuantizeConfig,
        params: Optional[Dict[str, TensorLike]] = None,
        dataset: Optional[Iterable] = None,
        *,
        entry: str = "main",
        target: str = "llvm",
        device: Optional[tvm.runtime.Device] = None,
        match_op_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.params = params
        self.dataset = dataset
        self.entry = entry
        self.target = target
        self.device = device
        self.match_op_names = match_op_names

    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:  # type: ignore[override] # noqa: D401
        return quantize(
            mod,
            self.params,
            dataset=self.dataset,
            config=self.config,
            entry=self.entry,
            target=self.target,
            device=self.device,
            match_op_names=self.match_op_names,
        )


__all__ = [
    "QuantizeConfig",
    "QuantizeContext",
    "QuantizePass",
    "Quantizer",
    "create_quantize_config",
    "qconfig",
    "quantize",
]
