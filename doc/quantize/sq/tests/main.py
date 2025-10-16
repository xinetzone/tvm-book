"""End-to-end check that the Relax quantization flow works on PyTorch ResNet18."""

from __future__ import annotations

import numpy as np
import torch
import torchvision
from torch import fx
import tvm
from tvm import relax
from tvm.relax.frontend import detach_params
from tvm.relax.frontend.torch.fx_translator import from_fx
from tvm.relax.testing import vm as rvm

from _calibrate import CalibrationMethod
from quantize import create_quantize_config, quantize as relax_quantize


def _as_nd(value):
    if isinstance(value, tvm.runtime.NDArray):
        return value
    return tvm.nd.array(value)


def build_resnet18_relax():
    """Load pretrained ResNet18 and convert to Relax IRModule."""
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
    example_input = torch.randn(1, 3, 224, 224)
    fx_mod = fx.symbolic_trace(model)

    input_info = [((1, 3, 224, 224), "float32")]
    mod = from_fx(fx_mod, input_info, keep_params_as_input=True)
    mod, params = detach_params(mod)
    return mod, params


def run_module(mod, params, data, target="llvm", device=None):
    """Build and execute a Relax module."""
    device = device or tvm.cpu()
    exec_mod = relax.build(mod, target=target)
    vm = rvm.VirtualMachine(exec_mod, device)
    if params:
        vm.set_input("main", **{k: _as_nd(v) for k, v in params.items()})
    return vm["main"](_as_nd(data)).numpy()


def main():
    mod, params = build_resnet18_relax()

    # Prepare calibration dataset (random inputs for demo purposes).
    calib_samples = [np.random.randn(1, 3, 224, 224).astype("float32") for _ in range(4)]

    config = create_quantize_config(
        calibrate_mode=CalibrationMethod.MIN_MAX,
        quantize_inputs=False,
        quantize_weights=True,
        quantize_outputs=True,
        weight_axis=0,
        per_channel_weights=True,
    )

    print("Running Relax quantization with min-max calibration ...")
    quantized_mod = relax_quantize(
        mod,
        params,
        dataset=calib_samples,
        config=config,
        target="llvm",
        device=tvm.cpu(),
    )

    # Evaluate float vs quantized outputs on a fresh sample.
    test_input = np.random.randn(1, 3, 224, 224).astype("float32")
    float_out = run_module(mod, params, test_input)
    quant_out = run_module(quantized_mod, params, test_input)

    abs_diff = np.max(np.abs(float_out - quant_out))
    rel_diff = abs_diff / max(np.max(np.abs(float_out)), 1e-6)
    print(f"Max absolute difference: {abs_diff:.6f}")
    print(f"Max relative difference: {rel_diff:.6f}")
    print("Float output shape:", float_out.shape)
    print("Quantized output shape:", quant_out.shape)


if __name__ == "__main__":
    main()
