import os
import scipy
import numpy as np
import tvm
import tvm.topi.testing
from tvm import relay
from tvm.contrib import graph_executor, utils
from tvm.relay.frontend.common import infer_type
from tvm.relay.build_module import bind_params_by_name
import onnx
import onnxruntime.backend
from onnx import TensorProto, helper, mapping, numpy_helper
from onnxruntime.quantization import CalibrationDataReader, quantize_static


def get_input_data_shape_dict(graph_def, input_data):
    """Get input data shape"""
    if isinstance(input_data, list):
        input_names = {}
        shape_dict = {}
        for i, _ in enumerate(input_data):
            input_names[i] = graph_def.graph.input[i].name
            input_ = input_data[i]

            if input_ is None or not hasattr(input_, "shape") or input_.shape == ():
                # Skip adding input shape data when the input data is None;
                # This is to enable optional arguments for onnx operators.
                continue

            elif isinstance(input_, list):
                shape_dict[input_names[i]] = (len(input_),)

            else:
                shape_dict[input_names[i]] = input_.shape

    else:
        input_names = graph_def.graph.input[0].name
        shape_dict = {input_names: input_data.shape}

    return input_names, shape_dict


def get_tvm_output_with_vm(
    graph_def,
    input_data,
    target,
    dev,
    opset=None,
    freeze_params=False,
    convert_config=None,
    validate_structural_equal=True,
):
    """Generic function to execute and get tvm output with vm executor"""
    if not isinstance(input_data, list):
        input_data = [input_data]
    _, shape_dict = get_input_data_shape_dict(graph_def, input_data)

    with tvm.testing.disable_span_filling():
        mod, params = relay.frontend.from_onnx(
            graph_def,
            shape_dict,
            opset=opset,
            freeze_params=freeze_params,
            convert_config=convert_config,
        )
        # handle the bfloat16 so we explicitly allocate
        # bfloat16 arrays as input
        for i, param in enumerate(mod["main"].params):
            if param.type_annotation.dtype == "bfloat16":
                input_data[i] = tvm.nd.empty(input_data[i].shape, "bfloat16").copyfrom(
                    input_data[i]
                )

    if validate_structural_equal:
        with tvm.testing.enable_span_filling():
            mod_with_span, _ = relay.frontend.from_onnx(
                graph_def,
                shape_dict,
                opset=opset,
                freeze_params=freeze_params,
                convert_config=convert_config,
            )
        assert tvm.ir.structural_equal(mod, mod_with_span)

    result = relay.create_executor("vm", mod=mod, device=dev, target=target).evaluate()(
        *input_data, **params
    )
    if isinstance(result, tvm.runtime.NDArray):
        return result.numpy()
    return [r.numpy() for r in result]


def get_tvm_output(
    graph_def,
    input_data,
    target,
    dev,
    output_shape=None,
    output_dtype="float32",
    opset=None,
    opt_level=1,
    convert_config=None,
):
    """Generic function to execute and get tvm output"""
    # TODO: Resolve the issues and remove the following lines
    input_names, shape_dict = get_input_data_shape_dict(graph_def, input_data)

    mod, params = relay.frontend.from_onnx(
        graph_def, shape_dict, opset=opset, convert_config=convert_config
    )

    with tvm.transform.PassContext(opt_level=opt_level):
        graph, lib, params = relay.build(mod, target, params=params)

    m = graph_executor.create(graph, lib, dev)
    # set inputs
    if isinstance(input_data, list):
        for i, _ in enumerate(input_names):
            # Its possible for some onnx inputs to not be needed in the tvm
            # module, confirm its present before setting.
            # pylint: disable=unnecessary-list-index-lookup
            m.set_input(input_names[i], tvm.nd.array(input_data[i].astype(input_data[i].dtype)))
    else:
        m.set_input(input_names, tvm.nd.array(input_data.astype(input_data.dtype)))

    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    if isinstance(output_shape, list):
        tvm_output_list = []
        for i, _ in enumerate(output_shape):
            tvm_output = m.get_output(i)
            tvm_output_list.append(tvm_output.numpy())
        return tvm_output_list
    else:
        tvm_output = m.get_output(0)
        return tvm_output.numpy()


def get_onnxruntime_output(model, inputs):
    """Generic function to generate onnxruntime output"""
    rep = onnxruntime.backend.prepare(model.SerializeToString(), "CPU")
    if isinstance(inputs, list) and len(inputs) == 1:
        inp = inputs[0]
    else:
        inp = inputs
    output = rep.run(inp)
    # Unpack output if there's only a single value.
    if len(output) == 1:
        output = output[0]
    return output


def verify_with_ort_with_inputs(
    model,
    inputs,
    out_shape=None,
    target=None,
    dev=None,
    use_vm=False,
    opset=None,
    freeze_params=False,
    dtype="float32",
    rtol=1e-5,
    atol=1e-5,
    apply_softmax=False,
    opt_level=1,
    convert_config=None,
):
    """verify_with_ort_with_inputs"""
    if opset is not None:
        model.opset_import[0].version = opset

    ort_out = get_onnxruntime_output(model, inputs)
    if use_vm:
        tvm_out = get_tvm_output_with_vm(
            model,
            inputs,
            target,
            dev,
            opset=opset,
            freeze_params=freeze_params,
            convert_config=convert_config,
        )
    else:
        tvm_out = get_tvm_output(
            model,
            inputs,
            target,
            dev,
            out_shape,
            dtype,
            opset=opset,
            opt_level=opt_level,
            convert_config=convert_config,
        )

    if not isinstance(tvm_out, list):
        tvm_out = [tvm_out]
    if not isinstance(ort_out, list):
        ort_out = [ort_out]
    for tvm_val, ort_val in zip(tvm_out, ort_out):
        if apply_softmax:
            ort_val = scipy.special.softmax(ort_val)
            tvm_val = scipy.special.softmax(tvm_val)
        tvm.testing.assert_allclose(ort_val, tvm_val, rtol=rtol, atol=atol)
        assert ort_val.dtype == tvm_val.dtype


def verify_with_ort(
    model,
    input_shapes,
    out_shape=None,
    target=None,
    dev=None,
    use_vm=False,
    opset=None,
    freeze_params=False,
    dtype="float32",
    rtol=1e-5,
    atol=1e-5,
):
    """verify_with_ort"""
    inputs = [np.random.uniform(size=ishape).astype(dtype) for ishape in input_shapes]
    verify_with_ort_with_inputs(
        model,
        inputs,
        out_shape=out_shape,
        target=target,
        dev=dev,
        use_vm=use_vm,
        opset=opset,
        freeze_params=freeze_params,
        dtype=dtype,
        rtol=rtol,
        atol=atol,
    )


def quantize_and_verify_with_ort(
    onnx_model, input_names, input_shapes, target, dev, rtol=1e-5, atol=1e-5
):
    """quantize_and_verify_with_ort"""
    input_arrays = [np.random.random(shape).astype("float32") for shape in input_shapes]

    class RandomDataReader(CalibrationDataReader):
        # pylint: disable=missing-class-docstring
        def __init__(self, n=10):
            input_dict = dict(zip(input_names, input_shapes))
            self.data = iter(
                [
                    {
                        name: np.random.random(shape).astype("float32")
                        for name, shape in input_dict.items()
                    }
                    for _ in range(n)
                ]
            )

        def get_next(self):
            return next(self.data, None)

    t_dir = tvm.contrib.utils.tempdir()
    model_fp32 = os.path.join(t_dir.temp_dir, "model.onnx")
    onnx.save_model(onnx_model, model_fp32)
    model_quant = os.path.join(t_dir.temp_dir, "model.quant.onnx")
    _ = quantize_static(  # pylint: disable=assignment-from-no-return
        model_fp32, model_quant, RandomDataReader()
    )
    # opt_level=1 will cause error with qnn lowering
    model = onnx.load(model_quant)
    verify_with_ort_with_inputs(
        model, input_arrays, opt_level=2, target=target, dev=dev, use_vm=True, rtol=rtol, atol=atol
    )


def make_constant_node(name, data_type, dims, vals):
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        value=helper.make_tensor(name=name, data_type=data_type, dims=dims, vals=vals),
    )
