import sys
# import numpy as np

import torch
# from torch.nn import functional as F
# import torchvision

import tvm
import tvm.testing
from tvm import relay
from tvm.contrib import graph_executor
# from tvm.contrib.nvcc import have_fp16
# from tvm.contrib import cudnn, utils

def assert_shapes_match(tru, est):
    """Verfiy whether the shapes are equal"""
    if tru.shape != est.shape:
        raise AssertionError("Output shapes {tru.shape} and {est.shape} don't match")

def verify_model(
    model_name,
    input_data=None,
    custom_convert_map=None,
    rtol=1e-5,
    atol=1e-5,
    expected_ops=None,
    kind="graph",
    check_correctness=True,
    cpu_only=False,
    validate_structural_equal=True,
):
    """Assert that the output of a compiled model matches with that of its
    baseline."""
    input_data = [] if input_data is None else input_data
    custom_convert_map = custom_convert_map or {}
    expected_ops = expected_ops or []
    if isinstance(model_name, str):
        baseline_model, baseline_input = load_model(model_name)
    elif isinstance(input_data, list):
        baseline_model = model_name
        baseline_input = input_data
    elif isinstance(input_data, torch.Tensor) or not input_data.shape:
        baseline_model = model_name
        baseline_input = [input_data]
    else:
        assert False, "Unexpected input format"
    if torch.cuda.is_available():
        if isinstance(baseline_model, torch.nn.Module):
            baseline_model = baseline_model.cuda()
        baseline_input = [inp.cuda() for inp in baseline_input]

    with torch.no_grad():
        baseline_outputs = baseline_model(*[input.clone() for input in baseline_input])

    if isinstance(baseline_outputs, tuple):
        baseline_outputs = tuple(out.cpu().numpy() for out in baseline_outputs)
    else:
        baseline_outputs = (baseline_outputs.cpu().numpy(),)

    trace = torch.jit.trace(baseline_model, [input.clone() for input in baseline_input])
    if isinstance(baseline_model, torch.nn.Module):
        trace = trace.float().eval()

        if torch.cuda.is_available():
            trace = trace.cuda()
        else:
            trace = trace.cpu()

    input_names = [f"input{idx}" for idx, _ in enumerate(baseline_input)]
    input_shapes = list(zip(input_names, [inp.shape for inp in baseline_input]))
    with tvm.testing.disable_span_filling():
        mod, params = relay.frontend.from_pytorch(trace, input_shapes, custom_convert_map)
    if validate_structural_equal:
        with tvm.testing.enable_span_filling():
            mod_with_span, _ = relay.frontend.from_pytorch(trace, input_shapes, custom_convert_map)
        assert tvm.ir.structural_equal(mod, mod_with_span, map_free_vars=True)

    for arg in mod["main"].params[: len(input_names)]:
        assert arg.name_hint in input_names
    compiled_input = dict(zip(input_names, [inp.clone().cpu().numpy() for inp in baseline_input]))

    targets = ["llvm"]
    if not cpu_only:
        targets.append("cuda")

    with tvm.transform.PassContext(opt_level=3):
        for target in targets:
            if not tvm.runtime.enabled(target):
                continue
            dev = tvm.device(target, 0)
            exe = relay.create_executor(
                kind, mod=mod, params=params, device=dev, target=target
            ).evaluate()
            result = exe(**compiled_input)
            if not isinstance(result, list):
                result = [result]

            for i, baseline_output in enumerate(baseline_outputs):
                output = result[i].numpy()

                assert_shapes_match(baseline_output, output)
                if check_correctness:
                    tvm.testing.assert_allclose(baseline_output, output, rtol=rtol, atol=atol)

    if expected_ops:

        def visit(op):
            if isinstance(op, tvm.ir.op.Op):
                if op.name in expected_ops:
                    expected_ops.remove(op.name)

        tvm.relay.analysis.post_order_visit(mod["main"].body, visit)

        if expected_ops:
            msg = "TVM Relay do not contain expected ops {}"
            raise AssertionError(msg.format(expected_ops))

    del model_name
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verify_model_with_input(
    test_func,
    input_data,
    *,
    input_dict=None,
    custom_convert_map=None,
    rtol=1e-5,
    atol=1e-5,
    assert_shape_only=False,
    validate_structural_equal=True,
):
    """Generic function to generate and compare Pytorch and TVM output"""
    input_dict = input_dict or {}
    custom_convert_map = custom_convert_map or {}
    baseline_outputs = test_func(*input_data)
    trace = torch.jit.trace(test_func, [input.clone() for input in input_data])
    input_names = [f"input{idx}" for idx, _ in enumerate(input_data)]
    input_shapes = list(zip(input_names, [inp.shape for inp in input_data]))
    with tvm.testing.disable_span_filling():
        mod, params = relay.frontend.from_pytorch(trace, input_shapes, custom_convert_map)
    if validate_structural_equal:
        with tvm.testing.enable_span_filling():
            mod_with_span, _ = relay.frontend.from_pytorch(trace, input_shapes, custom_convert_map)
        assert tvm.ir.structural_equal(mod, mod_with_span, map_free_vars=True)

    with tvm.transform.PassContext(opt_level=3):
        for target in ["llvm", "cuda"]:
            if not tvm.runtime.enabled(target):
                continue
            dev = tvm.device(target, 0)
            lib = relay.build(mod, target=target, params=params)
            relay_model = graph_executor.GraphModule(lib["default"](dev))
            for name, value in input_dict.items():
                relay_model.set_input(name, value)
            relay_model.run()

            compiled_output = relay_model.get_output(0).numpy()
            assert_shapes_match(baseline_outputs, compiled_output)
            if assert_shape_only is False:
                tvm.testing.assert_allclose(baseline_outputs, compiled_output, rtol=rtol, atol=atol)


def gen_ir_module(model, inputs, use_parser_friendly_name=False):
    """Helper function to generate IRModule with meaningful source information"""

    trace = torch.jit.trace(model, inputs)
    input_names = ["input{}".format(idx) for idx, _ in enumerate(inputs)]
    input_shapes = list(zip(input_names, [inp.shape for inp in inputs]))
    mod, _ = relay.frontend.from_pytorch(
        trace,
        input_shapes,
        use_parser_friendly_name=use_parser_friendly_name,
    )
    return mod
