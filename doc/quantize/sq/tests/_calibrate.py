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
"""Calibration utilities for model quantization"""

import numpy as np
import tvm
from tvm import relax
from tvm.relax import expr as relax_expr
from tvm.ir import IRModule
from tvm.relax.expr_functor import PyExprVisitor
from tvm.relax.utils import get_struct_info
from tvm.runtime import NDArray

class CalibrationMethod:
    """Enum class for calibration methods."""
    MIN_MAX = "min_max"
    KL_DIVERGENCE = "kl_divergence"
    ENTROPY = "entropy"
    GLOBAL_SCALE = "global_scale"

class CalibrateContext:
    """Context for model calibration."""
    def __init__(self, method=CalibrationMethod.MIN_MAX):
        self.method = method
        self.collectors = {}
        self.calibrated_params = {}

    def get_collector(self, name):
        """Get or create a collector for the given name."""
        if name not in self.collectors:
            if self.method == CalibrationMethod.MIN_MAX:
                self.collectors[name] = MinMaxCollector()
            elif self.method == CalibrationMethod.KL_DIVERGENCE:
                self.collectors[name] = KLCollector()
            elif self.method == CalibrationMethod.ENTROPY:
                self.collectors[name] = EntropyCollector()
            else:
                raise ValueError(f"Unsupported calibration method: {self.method}")
        return self.collectors[name]

class MinMaxCollector:
    """Collector for min-max calibration."""
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def collect(self, data):
        """Collect min and max values from data."""
        if not isinstance(data, np.ndarray) and not isinstance(data, NDArray):
            return

        data_np = data.asnumpy() if isinstance(data, NDArray) else data
        current_min = np.min(data_np)
        current_max = np.max(data_np)

        if self.min_val is None or current_min < self.min_val:
            self.min_val = current_min
        if self.max_val is None or current_max > self.max_val:
            self.max_val = current_max

    def get_params(self):
        """Get the calibrated parameters (scale and zero_point)."""
        return {"min_val": self.min_val, "max_val": self.max_val}

class KLCollector:
    """Collector for KL divergence calibration."""
    def __init__(self, num_bins=2048):
        self.histograms = []
        self.num_bins = num_bins

    def collect(self, data):
        """Collect histogram data."""
        if not isinstance(data, np.ndarray) and not isinstance(data, NDArray):
            return

        data_np = data.asnumpy() if isinstance(data, NDArray) else data
        # Flatten the data
        data_flat = data_np.flatten()
        # Compute histogram
        hist, bins = np.histogram(data_flat, bins=self.num_bins)
        self.histograms.append((hist, bins))

    def get_params(self):
        """Get the calibrated parameters using KL divergence."""
        if not self.histograms:
            return {"min_val": 0.0, "max_val": 1.0}

        # Combine all histograms
        total_hist = np.zeros(self.num_bins)
        for hist, _ in self.histograms:
            total_hist += hist

        # Find optimal threshold using KL divergence
        # This is a simplified version for demonstration
        min_val = min(bins[0] for _, bins in self.histograms)
        max_val = max(bins[-1] for _, bins in self.histograms)
        
        return {"min_val": min_val, "max_val": max_val}

class EntropyCollector:
    """Collector for entropy-based calibration."""
    def __init__(self, num_bins=2048):
        self.histograms = []
        self.num_bins = num_bins

    def collect(self, data):
        """Collect histogram data."""
        if not isinstance(data, np.ndarray) and not isinstance(data, NDArray):
            return

        data_np = data.asnumpy() if isinstance(data, NDArray) else data
        # Flatten the data
        data_flat = data_np.flatten()
        # Compute histogram
        hist, bins = np.histogram(data_flat, bins=self.num_bins)
        self.histograms.append((hist, bins))

    def get_params(self):
        """Get the calibrated parameters using entropy."""
        if not self.histograms:
            return {"min_val": 0.0, "max_val": 1.0}

        # Combine all histograms
        total_hist = np.zeros(self.num_bins)
        for hist, _ in self.histograms:
            total_hist += hist

        # Find optimal threshold using entropy
        # This is a simplified version for demonstration
        min_val = min(bins[0] for _, bins in self.histograms)
        max_val = max(bins[-1] for _, bins in self.histograms)
        
        return {"min_val": min_val, "max_val": max_val}

class CalibrationVisitor(PyExprVisitor):
    """Visitor for collecting calibration data."""
    def __init__(self, mod, params, calib_ctx):
        super().__init__()
        self.mod = mod
        self.params = params if params else {}
        self.calib_ctx = calib_ctx

    def visit_call(self, call):
        """Visit and collect data from function calls."""
        # Visit all arguments first
        for arg in call.args:
            self.visit_expr(arg)

        op_name = call.op.name
        
        # Collect data for convolution and dense layers
        if op_name in ["relax.nn.conv2d", "relax.nn.dense"]:
            # Get the input tensor
            input_arg = call.args[0]
            if isinstance(input_arg, relax_expr.Var):
                # Generate a unique name for this operation
                op_name = f"{op_name}_{input_arg.name_hint}"
                # Get collector for this operation
                collector = self.calib_ctx.get_collector(op_name)
                
                # In a real scenario, we would need to run the model with calibration data
                # and collect the actual tensor values. This is a simplified version.
                # For demonstration purposes, we'll just create dummy data.
                if input_arg.name_hint in self.params:
                    data = self.params[input_arg.name_hint]
                    collector.collect(data)

        super().visit_call(call)

# Helper function to create calibration context
def create_calibrate_context(method=CalibrationMethod.MIN_MAX):
    """Create a calibration context with the specified method."""
    return CalibrateContext(method)

def calibrate_model(mod, params, calib_ctx):
    """Calibrate a model to collect quantization parameters.

    Parameters
    ----------
    mod : tvm.IRModule
        The Relay module to calibrate.
    params : dict
        The parameters of the module.
    calib_ctx : CalibrateContext
        The calibration context.

    Returns
    -------
    calibrated_params : dict
        The calibrated quantization parameters.
    """
    # Create calibration visitor and visit the module
    visitor = CalibrationVisitor(mod, params, calib_ctx)
    for gv, func in mod.functions.items():
        if isinstance(func, relax_expr.Function):
            visitor.visit_expr(func)

    # Extract calibrated parameters
    calibrated_params = {}
    for name, collector in calib_ctx.collectors.items():
        params = collector.get_params()
        calibrated_params[f"{name}_min"] = tvm.ir.const(params["min_val"])
        calibrated_params[f"{name}_max"] = tvm.ir.const(params["max_val"])

    # Store calibrated parameters in the context
    calib_ctx.calibrated_params = calibrated_params
    
    return calibrated_params

def calculate_scale_zero_point(min_val, max_val, dtype):
    """Calculate scale and zero_point for quantization.

    Parameters
    ----------
    min_val : float
        Minimum value of the tensor.
    max_val : float
        Maximum value of the tensor.
    dtype : str
        Target data type.

    Returns
    -------
    scale : float
        The scale factor.
    zero_point : int
        The zero point.
    """
    # Get min and max values for the target dtype
    if dtype == "int8":
        qmin, qmax = -128, 127
    elif dtype == "uint8":
        qmin, qmax = 0, 255
    else:
        raise ValueError(f"Unsupported dtype for quantization: {dtype}")

    # Calculate scale and zero_point
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale
    
    # Clip zero_point to the valid range
    zero_point = np.clip(zero_point, qmin, qmax)
    zero_point = int(round(zero_point))
    
    return scale, zero_point

__all__ = [
    "CalibrationMethod",
    "CalibrateContext",
    "create_calibrate_context",
    "calibrate_model",
    "calculate_scale_zero_point"
]