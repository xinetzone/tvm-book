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
"""Relay model quantization framework"""

import tvm
from tvm import relax
from tvm.relax import expr as relax_expr
from tvm.relax.op import qdq
from tvm.ir import IRModule
from tvm.relax.transform import FunctionPass, ModulePass
from tvm.relax.expr_functor import PyExprVisitor, PyExprMutator
from tvm.relax.utils import get_struct_info
from . import _ffi_api

# Import calibration module
from ._calibrate import (  # pylint: disable=wrong-import-position
    calibrate_model,
    create_calibrate_context,
    CalibrationMethod,
)

class QuantizeConfig:
    """Configuration for model quantization.

    Parameters
    ----------
    calibrate_mode : str
        Calibration mode for quantization. Can be 'global_scale', 'kl_divergence', 'entropy' or 'min_max'.
    global_scale : float, optional
        Global scale factor when calibrate_mode is 'global_scale'. Default is 8.0.
    nbit_input : int
        Number of bits for input quantization. Default is 8.
    nbit_weight : int
        Number of bits for weight quantization. Default is 8.
    dtype_input : str
        Data type for quantized input. Default is 'int8'.
    dtype_weight : str
        Data type for quantized weight. Default is 'int8'.
    quantize_type : str
        Quantization type. Can be 'int' or 'uint'. Default is 'int'.
    axis : int
        Channel axis for per-channel quantization. Default is -1.
    skip_dense_layer : bool
        Whether to skip dense layers. Default is False.
    skip_conv_layers : bool
        Whether to skip convolution layers. Default is False.
    """
    def __init__(self,
                 calibrate_mode="global_scale",
                 global_scale=8.0,
                 nbit_input=8,
                 nbit_weight=8,
                 dtype_input="int8",
                 dtype_weight="int8",
                 quantize_type="int",
                 axis=-1,
                 skip_dense_layer=False,
                 skip_conv_layers=False):
        self.calibrate_mode = calibrate_mode
        self.global_scale = global_scale
        self.nbit_input = nbit_input
        self.nbit_weight = nbit_weight
        self.dtype_input = dtype_input
        self.dtype_weight = dtype_weight
        self.quantize_type = quantize_type
        self.axis = axis
        self.skip_dense_layer = skip_dense_layer
        self.skip_conv_layers = skip_conv_layers

class QuantizeContext:
    """Context manager for quantization configuration."""
    _current = None

    def __init__(self, config):
        self.config = config
        self._old_context = None

    def __enter__(self):
        self._old_context = QuantizeContext._current
        QuantizeContext._current = self
        return self

    def __exit__(self, ptype, value, trace):
        QuantizeContext._current = self._old_context

    @staticmethod
    def current():
        """Get the current quantization context."""
        return QuantizeContext._current

# Helper function to create quantization configuration
def qconfig(
    calibrate_mode="global_scale",
    global_scale=8.0,
    nbit_input=8,
    nbit_weight=8,
    dtype_input="int8",
    dtype_weight="int8",
    quantize_type="int",
    axis=-1,
    skip_dense_layer=False,
    skip_conv_layers=False,
):
    """Create a quantization configuration."""
    config = QuantizeConfig(
        calibrate_mode=calibrate_mode,
        global_scale=global_scale,
        nbit_input=nbit_input,
        nbit_weight=nbit_weight,
        dtype_input=dtype_input,
        dtype_weight=dtype_weight,
        quantize_type=quantize_type,
        axis=axis,
        skip_dense_layer=skip_dense_layer,
        skip_conv_layers=skip_conv_layers,
    )
    return QuantizeContext(config)

class Quantizer(PyExprMutator):
    """Quantization expression mutator."""
    def __init__(self, mod, params, config):
        super().__init__()
        self.mod = mod
        self.params = params
        self.config = config
        self.calibrated_params = {}
        self.visited = set()

    def quantize_weight(self, weight, name):
        """Quantize weight parameter."""
        # Calculate scale and zero_point
        if self.config.calibrate_mode == "global_scale":
            scale = self.config.global_scale
            zero_point = 0
        else:
            # For other calibration modes, use min/max values
            min_val = weight.min()
            max_val = weight.max()
            qmin = tvm.runtime.nd.min_value(self.config.dtype_weight)
            qmax = tvm.runtime.nd.max_value(self.config.dtype_weight)
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - min_val / scale
            zero_point = tvm.ir.const(tvm.runtime.nd.clip(zero_point, qmin, qmax).asnumpy())
            scale = tvm.ir.const(scale)

        # Create quantized weight
        quantized_weight = qdq.quantize(
            weight,
            scale,
            zero_point,
            self.config.axis,
            self.config.dtype_weight
        )
        
        # Store scale and zero_point for dequantization
        self.calibrated_params[f"{name}_scale"] = scale
        self.calibrated_params[f"{name}_zero_point"] = zero_point
        
        return quantized_weight

    def visit_call(self, call):
        """Visit and mutate function calls."""
        # Skip if already visited
        if call in self.visited:
            return call
        self.visited.add(call)

        # Get original call
        new_call = super().visit_call(call)
        op_name = new_call.op.name
        new_args = list(new_call.args)

        # Quantize weights for convolution and dense layers
        if op_name == "relax.nn.conv2d" and not self.config.skip_conv_layers:
            # Weight is usually the second argument
            if len(new_args) > 1:
                weight_arg = new_args[1]
                if isinstance(weight_arg, relax_expr.Var) and weight_arg.name_hint in self.params:
                    # Quantize weight
                    weight = self.params[weight_arg.name_hint]
                    new_args[1] = self.quantize_weight(weight, weight_arg.name_hint)

        elif op_name == "relax.nn.dense" and not self.config.skip_dense_layer:
            # Weight is usually the second argument
            if len(new_args) > 1:
                weight_arg = new_args[1]
                if isinstance(weight_arg, relax_expr.Var) and weight_arg.name_hint in self.params:
                    # Quantize weight
                    weight = self.params[weight_arg.name_hint]
                    new_args[1] = self.quantize_weight(weight, weight_arg.name_hint)

        # Return modified call
        return relax_expr.Call(new_call.op, new_args, new_call.attrs, new_call.type_args)

    def transform(self):
        """Transform the module for quantization."""
        # Apply calibration if needed
        if self.config.calibrate_mode != "global_scale":
            # Create calibration context
            calib_ctx = create_calibrate_context(self.config.calibrate_mode)
            # Calibrate model to get scale factors
            calibrated_params = calibrate_model(self.mod, self.params, calib_ctx)
            self.calibrated_params.update(calibrated_params)

        # Mutate all functions in the module
        for gv, func in self.mod.functions.items():
            if isinstance(func, relax_expr.Function):
                self.mod[gv] = self.visit_expr(func)

        # Add calibrated parameters to the module
        for name, param in self.calibrated_params.items():
            if name not in self.mod:
                self.mod[name] = param

        return self.mod

@tvm.transform.register_transform_pass(name="Quantize", opt_level=0)
class QuantizePass(ModulePass):
    """A module pass for model quantization."""
    def __init__(self, config):
        self.config = config

    def transform_module(self, mod, ctx):
        """Transform the module."""
        # Get parameters from the module
        params = {}
        for gv, func in mod.functions.items():
            if isinstance(func, relax_expr.Constant):
                params[gv.name_hint] = func

        # Create quantizer and transform the module
        quantizer = Quantizer(mod, params, self.config)
        return quantizer.transform()

def quantize(mod, params=None):
    """Quantize a Relay module.

    Parameters
    ----------
    mod : tvm.IRModule
        The Relay module to quantize.
    params : dict, optional
        The parameters of the module.

    Returns
    -------
    quantized_mod : tvm.IRModule
        The quantized module.
    """
    # Get current quantization configuration
    ctx = QuantizeContext.current()
    if ctx is None:
        # Use default configuration if no context is set
        config = QuantizeConfig()
    else:
        config = ctx.config

    # Create quantization pass and apply it
    quantize_pass = QuantizePass(config)
    quantized_mod = quantize_pass(mod)

    return quantized_mod

__all__ = ["qconfig", "quantize", "QuantizeConfig", "QuantizeContext"]