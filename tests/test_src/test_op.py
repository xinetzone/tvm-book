
def _test_onnx_op_elementwise(
    target, dev, inshape, outfunc, npargs, dtype, opname, kwargs, opset=None, verify=True
):
    indata = np.random.uniform(-1, 1, size=inshape).astype(dtype)
    outdata = outfunc(indata, **npargs)

    y = helper.make_node(opname, ["in"], ["out"], **kwargs)

    ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]

    graph = helper.make_graph(
        [y],
        opname + "_test",
        inputs=[helper.make_tensor_value_info("in", ONNX_DTYPE, list(indata.shape))],
        outputs=[helper.make_tensor_value_info("out", ONNX_DTYPE, list(outdata.shape))],
    )

    model = helper.make_model(graph, producer_name=opname + "_test")
    if verify:
        verify_with_ort_with_inputs(
            model, [indata], [outdata.shape], opset=opset, dtype=dtype, target=target, dev=dev
        )
    else:
        get_tvm_output(
            model,
            [indata],
            target,
            dev,
            [outdata.shape],
            dtype,
            opset=opset,
            opt_level=3,
        )


@tvm.testing.parametrize_targets
def test_floor(target, dev):
    _test_onnx_op_elementwise(target, dev, (2, 4, 5, 6), np.floor, {}, "float32", "Floor", {})


@tvm.testing.parametrize_targets
def test_ceil(target, dev):
    _test_onnx_op_elementwise(target, dev, (2, 4, 5, 6), np.ceil, {}, "float32", "Ceil", {})


@tvm.testing.parametrize_targets
def test_clip(target, dev):
    """test_clip"""
    _test_onnx_op_elementwise(
        target,
        dev,
        (2, 4, 5, 6),
        np.clip,
        {"a_min": -1.0, "a_max": 1.0},
        "float32",
        "Clip",
        {"min": -1.0, "max": 1.0},
        opset=6,
    )

    _test_onnx_op_elementwise(
        target,
        dev,
        (2, 4, 5, 6),
        np.clip,
        {"a_min": -np.inf, "a_max": 1.0},
        "float32",
        "Clip",
        {"max": 1.0},
        opset=6,
    )

    _test_onnx_op_elementwise(
        target,
        dev,
        (2, 4, 5, 6),
        np.clip,
        {"a_min": -1.0, "a_max": np.inf},
        "float32",
        "Clip",
        {"min": -1.0},
        opset=6,
    )


@tvm.testing.parametrize_targets
def test_clip_min_max_as_inputs(target, dev):
    """test_clip_min_max_as_inputs"""
    input_shape = (2, 4, 5, 6)
    nodes = [
        make_constant_node("min", onnx.TensorProto.FLOAT, (), [0.0]),
        make_constant_node("max", onnx.TensorProto.FLOAT, (), [6.0]),
    ]
    input_names = ["in", "min", "max"]
    nodes.append(helper.make_node("Clip", inputs=input_names, outputs=["out"]))
    graph = helper.make_graph(
        nodes,
        "clip_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(input_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(input_shape))],
    )
    model = helper.make_model(graph, producer_name="clip_test")

    verify_with_ort(model, [input_shape], out_shape=[input_shape], target=target, dev=dev)


@tvm.testing.parametrize_targets
def test_round(target, dev):
    _test_onnx_op_elementwise(target, dev, (2, 4, 5, 6), np.round, {}, "float32", "Round", {})
    _test_onnx_op_elementwise(
        target, dev, (2, 4, 5, 6), np.round, {}, "float64", "Round", {}, verify=False
    )  # TODO: enable verification once ORT supports float64


def _test_finite_ops(target, dev, inshape, outfunc, npargs, dtype, opname, kwargs):
    indata = np.random.choice(a=[np.nan, np.inf, -np.inf, 0.5, 1.0, 0], size=inshape).astype(dtype)

    outdata = outfunc(indata, **npargs)
    y = helper.make_node(opname, ["in"], ["out"], **kwargs)

    graph = helper.make_graph(
        [y],
        opname + "_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(indata.shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.BOOL, list(outdata.shape))],
    )

    model = helper.make_model(graph, producer_name=opname + "_test")
    verify_with_ort_with_inputs(
        model, [indata], [outdata.shape], dtype=dtype, target=target, dev=dev
    )


@tvm.testing.parametrize_targets
def test_isinf(target, dev):
    _test_finite_ops(target, dev, (2, 4, 5, 6), np.isinf, {}, "float32", "IsInf", {})


@tvm.testing.parametrize_targets
def test_isnan(target, dev):
    """test_isnan"""
    _test_finite_ops(target, dev, (2, 4, 5, 6), np.isnan, {}, "float32", "IsNaN", {})


@tvm.testing.parametrize_targets
def test_gather_nd(target, dev):
    """test_gather_nd"""

    def verify_gather_nd(in_shape, indices, out_shape, dtype="float32", batch_dims=0, opset=11):
        x = np.random.uniform(size=in_shape).astype(dtype)
        indices = np.array(indices, dtype="int64")

        y = helper.make_node("GatherND", ["in", "indices"], ["out"])

        if opset >= 12:
            batch_dims_attr = helper.make_attribute("batch_dims", batch_dims)
            y.attribute.append(batch_dims_attr)

        graph = helper.make_graph(
            [y],
            "gather_test",
            inputs=[
                helper.make_tensor_value_info(
                    "in", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], list(in_shape)
                ),
                helper.make_tensor_value_info("indices", TensorProto.INT64, list(indices.shape)),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "out", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], list(out_shape)
                )
            ],
        )
        model = helper.make_model(graph, producer_name="gather_test")
        verify_with_ort_with_inputs(
            model, [x, indices], [out_shape], opset=opset, target=target, dev=dev
        )

    verify_gather_nd([2, 2], [[0, 0], [1, 1]], [2], "int32")
    verify_gather_nd([2, 2], [[1], [0]], [2, 2])
    verify_gather_nd([2, 2, 2], [[0, 1], [1, 0]], [2, 2])
    verify_gather_nd([2, 2, 2], [[[0, 1]], [[1, 0]]], [2, 1, 2])

    if is_version_greater_than("1.6.0"):
        verify_gather_nd([2, 2, 2], [[1], [0]], [2, 2], batch_dims=1, opset=12)
        verify_gather_nd(
            (3, 2, 2, 3, 4),
            np.random.randint(low=0, high=2, size=(3, 2, 3), dtype="int64"),
            (3, 2),
            batch_dims=2,
            opset=12,
        )


@tvm.testing.parametrize_targets
def test_onehot(target, dev):
    """test_onehot"""
    indices_shape = [10]
    indices_array = np.random.randint(low=0, high=9, size=indices_shape, dtype="int32")
    depth = 10
    values = np.asarray([0, 1]).astype("int32")
    out_np = np.eye(depth)[indices_array.reshape(-1)]

    onehot_node = helper.make_node("OneHot", ["indices", "depth", "values"], ["out"])

    graph = helper.make_graph(
        [onehot_node],
        "onehot_test",
        inputs=[
            helper.make_tensor_value_info("indices", TensorProto.INT32, indices_shape),
            helper.make_tensor_value_info("depth", TensorProto.INT32, [1]),
            helper.make_tensor_value_info("values", TensorProto.INT32, values.shape),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.INT32, out_np.shape)],
    )

    model = helper.make_model(graph, producer_name="onehot_test")

    # TODO(jwfromm): Replace test against np with test against onnxrt once we update versions.
    tvm_out = get_tvm_output_with_vm(
        model, [indices_array, np.array([depth]).astype("int32"), values], target, dev
    )
    tvm.testing.assert_allclose(out_np, tvm_out, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets
def test_gemm(target, dev):
    """test_gemm"""

    def verify_gemm(a_shape, b_shape, c_shape=None, freeze_params=False, dtype="float32"):
        out_shape = [a_shape[0], b_shape[1]]
        a_array = np.random.uniform(size=a_shape).astype(dtype)
        b_array = np.random.uniform(size=b_shape).astype(dtype)
        input_names = ["a", "b"]
        ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
        input_nodes = [
            helper.make_tensor_value_info("a", ONNX_DTYPE, list(a_shape)),
            helper.make_tensor_value_info("b", ONNX_DTYPE, list(b_shape)),
        ]
        input_values = [a_array, b_array]
        if c_shape is not None:
            c_array = np.random.uniform(size=c_shape).astype(dtype)
            input_names.append("c")
            input_nodes.append(helper.make_tensor_value_info("c", ONNX_DTYPE, list(c_shape)))
            input_values.append(c_array)

        gemm_node = helper.make_node("Gemm", input_names, ["out"])

        graph = helper.make_graph(
            [gemm_node],
            "gemm_test",
            inputs=input_nodes,
            outputs=[helper.make_tensor_value_info("out", ONNX_DTYPE, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="gemm_test")
        atol = 1e-5
        rtol = 1e-5
        if dtype == "float16":
            atol = 1e-3
            rtol = 1e-3
        verify_with_ort_with_inputs(
            model,
            input_values,
            freeze_params=freeze_params,
            dtype=dtype,
            atol=atol,
            rtol=rtol,
            target=target,
            dev=dev,
        )

    verify_gemm(a_shape=(4, 3), b_shape=(3, 4))
    verify_gemm(a_shape=(4, 3), b_shape=(3, 4), c_shape=(4,))
    verify_gemm(a_shape=(4, 3), b_shape=(3, 4), c_shape=(4,), freeze_params=True)
    verify_gemm(a_shape=(4, 3), b_shape=(3, 4), c_shape=(4,), freeze_params=True, dtype="float16")


@tvm.testing.parametrize_targets
def test_matmul(target, dev):
    """test_matmul"""

    def test_one_matmul(a_shape, b_shape):
        out_shape = np.matmul(np.zeros(a_shape), np.zeros(b_shape)).shape

        a_array = np.random.uniform(size=a_shape).astype("float32")
        b_array = np.random.uniform(size=b_shape).astype("float32")

        mul_node = helper.make_node("MatMul", ["a", "b"], ["out"])

        graph = helper.make_graph(
            [mul_node],
            "matmul_test",
            inputs=[
                helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
                helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="matmul_test")
        verify_with_ort_with_inputs(model, [a_array, b_array], target=target, dev=dev)

    test_one_matmul((4, 3), (3, 4))
    test_one_matmul((3,), (3, 1))
    test_one_matmul((1, 3), (3,))
    test_one_matmul((3,), (3,))


@tvm.testing.parametrize_targets
def test_batch_matmul(target, dev):
    """test_batch_matmul"""

    def verify_batch_matmul(a_shape, b_shape, out_shape, convert_config=None):
        a_array = np.random.uniform(size=a_shape).astype("float32")
        b_array = np.random.uniform(size=b_shape).astype("float32")

        mul_node = helper.make_node("MatMul", ["a", "b"], ["out"])

        graph = helper.make_graph(
            [mul_node],
            "matmul_test",
            inputs=[
                helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
                helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)],
        )

        model = helper.make_model(graph, producer_name="matmul_test")
        verify_with_ort_with_inputs(
            model,
            [a_array, b_array],
            use_vm=True,
            target=target,
            dev=dev,
            convert_config=convert_config,
        )

    verify_batch_matmul((2, 3, 4, 3), (2, 3, 3, 4), (2, 3, 4, 4))
    verify_batch_matmul((2, 4, 3), (3, 4), (2, 4, 4))
    verify_batch_matmul((2, 3, 4, 3), (3, 4), (2, 3, 4, 4))
    # Test implicit broadcasting.
    verify_batch_matmul((4, 3), (2, 3, 4), (2, 4, 4))
    verify_batch_matmul((2, 4, 3), (1, 3, 4), (2, 4, 4))
    verify_batch_matmul((1, 4, 3), (2, 3, 4), (2, 4, 4))
    verify_batch_matmul((4, 32, 16), (16, 32), (4, 32, 32))
    verify_batch_matmul((4, 32, 16, 32), (32, 16), (4, 32, 16, 16))
    verify_batch_matmul((4, 32, 16, 32), (1, 32, 32, 16), (4, 32, 16, 16))
    verify_batch_matmul((4, 1, 16, 32), (1, 32, 32, 16), (4, 32, 16, 16))
    # Test transb=False
    verify_batch_matmul(
        (2, 3, 4, 3),
        (2, 3, 3, 4),
        (2, 3, 4, 4),
        convert_config={"use_nt_batch_matmul": False},
    )


@tvm.testing.parametrize_targets
def test_use_nt_batch_matmul(target, dev):
    """test_use_nt_batch_matmul"""
    a_shape = (2, 3, 4)
    b_shape = (2, 4, 3)
    out_shape = [2, 3, 3]
    a_array = np.random.uniform(size=a_shape).astype("float32")
    b_array = np.random.uniform(size=b_shape).astype("float32")

    for use_nt_batch_matmul in [True, False]:
        mul_node = helper.make_node("MatMul", ["a", "b"], ["out"])

        graph = helper.make_graph(
            [mul_node],
            "matmul_test",
            inputs=[
                helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
                helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="matmul_test")
        _, shape_dict = get_input_data_shape_dict(model, [a_array, b_array])

        mod, _ = relay.frontend.from_onnx(
            model, shape_dict, convert_config={"use_nt_batch_matmul": use_nt_batch_matmul}
        )
        has_transpose_op = "transpose" in str(mod)
        # use_nt_batch_matmul implies, TVM converts qualified onnx `matmul`
        # to `transpose(weight) + nn.batch_matmul_NT`, otherwise to `nn.batch_matmul`
        assert has_transpose_op == use_nt_batch_matmul


@tvm.testing.parametrize_targets
def test_matmulinteger16(target, dev):
    """test_matmulinteger16"""

    def verify_matmulinteger16(a_shape, b_shape, out_shape):
        a_dtype = "int16"
        b_dtype = "int16"
        low = np.iinfo(np.int16).min
        high = np.iinfo(np.int16).max

        a_proto = TensorProto.INT16
        b_proto = TensorProto.INT16
        out_proto = TensorProto.INT32
        a_array = np.random.randint(low, high, size=a_shape).astype(a_dtype)
        b_array = np.random.randint(low, high, size=b_shape).astype(b_dtype)

        mul_node = helper.make_node("MatMulInteger16", ["a", "b"], ["out"], domain="com.microsoft")

        graph = helper.make_graph(
            [mul_node],
            "matmuli16_test",
            inputs=[
                helper.make_tensor_value_info("a", a_proto, list(a_shape)),
                helper.make_tensor_value_info("b", b_proto, list(b_shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", out_proto, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="matmuli16_test")
        verify_with_ort_with_inputs(model, [a_array, b_array], target=target, dev=dev)

    # 2D computation to verify matmul op
    verify_matmulinteger16((4, 3), (3, 4), (4, 4))
    verify_matmulinteger16((5, 7), (7, 8), (5, 8))
    # Verify 3D matmul using batch_matmul op
    verify_matmulinteger16((2, 4, 3), (1, 3, 4), (2, 4, 4))
    verify_matmulinteger16((1, 4, 3), (2, 3, 4), (2, 4, 4))
    # Test implicit broadcasting
    verify_matmulinteger16((2, 3, 5, 3), (2, 3, 3, 5), (2, 3, 5, 5))
    verify_matmulinteger16((2, 7, 3), (3, 7), (2, 7, 7))
    verify_matmulinteger16((2, 3, 4, 3), (3, 4), (2, 3, 4, 4))


def verify_simple_dynamic_model(a_shape, b_shape, target, dev):
    """verify_simple_dynamic_model"""

    def verify_model(model, a_shape, b_shape):
        a_array = np.random.uniform(size=a_shape).astype("float32")
        b_array = np.random.uniform(size=b_shape).astype("float32")
        # matmul
        out_np = np.matmul(a_array, b_array)
        # relu
        out_np[out_np < 0] = 0

        tvm_out = model(a_array, b_array).numpy()
        tvm.testing.assert_allclose(out_np, tvm_out, rtol=1e-5, atol=1e-5)

    mul_node = helper.make_node("MatMul", ["a", "b"], ["out"])
    relu_node = helper.make_node("Relu", ["out"], ["relu"])

    a_array = np.random.uniform(size=a_shape).astype("float32")
    b_array = np.random.uniform(size=b_shape).astype("float32")
    # matmul
    out_np = np.matmul(a_array, b_array)

    graph = helper.make_graph(
        [mul_node, relu_node],
        "matmul_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape)),
        ],
        outputs=[helper.make_tensor_value_info("relu", TensorProto.FLOAT, list(out_np.shape))],
    )

    model = helper.make_model(graph, producer_name="matmul_test")

    a_anys = [relay.Any()] * len(a_shape)
    b_anys = [relay.Any()] * len(b_shape)

    mod, _ = relay.frontend.from_onnx(model, {"a": a_anys, "b": b_anys})
    model = relay.create_executor("vm", mod=mod, device=dev, target=target).evaluate()
    verify_model(model, a_shape, b_shape)
    verify_model(model, [a * 2 for a in a_shape], [b * 2 for b in b_shape])
    verify_model(model, [a * 3 for a in a_shape], [b * 3 for b in b_shape])


# TODO(mbrookhart, electriclilies): Add CUDA as a target once batch matmul is fixed
@tvm.testing.parametrize_targets("llvm")
def test_batch_matmul_dynamic_model(target, dev):
    verify_simple_dynamic_model((2, 3, 4, 3), (2, 3, 3, 4), target, dev)
    verify_simple_dynamic_model((2, 4, 3), (3, 4), target, dev)
    verify_simple_dynamic_model((2, 3, 4, 3), (3, 4), target, dev)


@tvm.testing.parametrize_targets
def test_lrn(target, dev):
    """test_lrn"""

    def verify_lrn(shape, nsize, dtype, alpha=None, beta=None, bias=None):
        in_array = np.random.uniform(size=shape).astype(dtype)

        if alpha is None and beta is None and bias is None:
            alpha = 0.0001
            beta = 0.75
            bias = 1.0
            node = onnx.helper.make_node("LRN", inputs=["in"], outputs=["out"], size=nsize)
        else:
            node = onnx.helper.make_node(
                "LRN", inputs=["in"], outputs=["out"], alpha=alpha, beta=beta, bias=bias, size=nsize
            )

        graph = helper.make_graph(
            [node],
            "lrn_test",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(shape))],
        )
        model = helper.make_model(graph, producer_name="lrn_test")
        verify_with_ort_with_inputs(model, [in_array], target=target, dev=dev)

    verify_lrn((5, 5, 5, 5), 3, "float32")
    verify_lrn((5, 5, 5, 5), 3, "float32", alpha=0.0002, beta=0.5, bias=2.0)


@tvm.testing.parametrize_targets
def test_instance_norm(target, dev):
    """test_instance_norm"""

    def verify_instance_norm(shape, axis=1):
        x = np.random.randn(*shape).astype(np.float32)
        gamma = np.random.randn(shape[1]).astype(np.float32)
        beta = np.random.randn(shape[1]).astype(np.float32)
        epsilon = 1e-5

        node = onnx.helper.make_node(
            "InstanceNormalization",
            inputs=["x", "gamma", "beta"],
            outputs=["y"],
            epsilon=epsilon,
        )
        graph = helper.make_graph(
            [node],
            "instance_norm_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(shape)),
                helper.make_tensor_value_info("gamma", TensorProto.FLOAT, (shape[1],)),
                helper.make_tensor_value_info("beta", TensorProto.FLOAT, (shape[1],)),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(shape))],
        )
        model = helper.make_model(graph, producer_name="instance_norm_test")
        verify_with_ort_with_inputs(
            model, [x, gamma, beta], out_shape=[shape], target=target, dev=dev
        )

    verify_instance_norm((2, 3, 4, 5))
    verify_instance_norm((32, 64, 80, 64))
    verify_instance_norm((8, 6, 5))
    verify_instance_norm((8, 7, 6, 5, 4))


@tvm.testing.parametrize_targets
def test_upsample_nearest(target, dev):
    """test_upsample_nearest"""
    scale = 2
    in_shape = (1, 1, 3, 3)
    out_shape = (1, 1, 3 * scale, 3 * scale)
    y = helper.make_node("Upsample", ["in"], ["out"], mode="nearest", scales=[1.0, 1.0, 2.0, 2.0])

    in_array = np.random.uniform(size=in_shape).astype(np.float32)

    graph = helper.make_graph(
        [y],
        "upsample_nearest_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="upsample_nearest_test")
    verify_with_ort_with_inputs(model, [in_array], [out_shape], opset=7, target=target, dev=dev)


@tvm.testing.parametrize_targets
def test_upsample3d_nearest(target, dev):
    """test_upsample3d_nearest"""
    scale = 2
    in_shape = (1, 1, 3, 3, 3)
    out_shape = (1, 1, 3 * scale, 3 * scale, 3 * scale)
    y = helper.make_node(
        "Upsample", ["in"], ["out"], mode="nearest", scales=[1.0, 1.0, 2.0, 2.0, 2.0]
    )

    in_array = np.random.uniform(size=in_shape).astype(np.float32)

    graph = helper.make_graph(
        [y],
        "upsample_nearest_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="upsample_nearest_test")
    # Upsample is deprecated after opset 9
    verify_with_ort_with_inputs(model, [in_array], [out_shape], opset=7, target=target, dev=dev)


@tvm.testing.parametrize_targets
def test_upsample_bilinear(target, dev):
    """test_upsample_bilinear"""
    scale = 2
    in_shape = (1, 1, 3, 3)
    out_shape = (1, 1, 3 * scale, 3 * scale)
    y = helper.make_node("Upsample", ["in"], ["out"], mode="linear", scales=[1.0, 1.0, 2.0, 2.0])

    in_array = np.random.uniform(size=in_shape).astype(np.float32)

    graph = helper.make_graph(
        [y],
        "upsample_bilinear_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="upsample_bilinear_test")
    verify_with_ort_with_inputs(model, [in_array], [out_shape], opset=7, target=target, dev=dev)


@tvm.testing.parametrize_targets
def test_upsample3d_trilinear(target, dev):
    """test_upsample3d_trilinear"""
    scale = 2
    in_shape = (1, 1, 3, 3, 3)
    out_shape = (1, 1, 3 * scale, 3 * scale, 3 * scale)
    y = helper.make_node("Upsample", ["in", "scales"], ["out"], mode="linear")
    scales = [1.0, 1.0, 2.0, 2.0, 2.0]
    in_array = np.random.uniform(size=in_shape).astype(np.float32)
    out_array = tvm.topi.testing.resize3d_python(
        in_array,
        (scale, scale, scale),
        "NCDHW",
        "linear",
        coordinate_transformation_mode="asymmetric",
    )

    ref_array = np.array(scales)
    ref_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["scales"],
        value=onnx.helper.make_tensor(
            name="const_tensor",
            data_type=TensorProto.FLOAT,
            dims=ref_array.shape,
            vals=ref_array.flatten().astype(float),
        ),
    )

    graph = helper.make_graph(
        [ref_node, y],
        "upsample_trilinear_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(in_shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="upsample_trilinear_test")
    # TODO(jwfromm): Trilinear upsampling not supported in 1.0.0 onnxruntime.
    # Replace topi comparison with verify_with_ort once we update.
    tvm_out = get_tvm_output(model, in_array, target, dev, out_shape, "float32")
    tvm.testing.assert_allclose(out_array, tvm_out, rtol=1e-5, atol=1e-5)


# TODO: Fix softmax with dynamic input on cuda and enable this test
@tvm.testing.known_failing_targets("cuda")
@tvm.testing.parametrize_targets
def test_softmax(target, dev):
    """test_softmax"""

    def verify_softmax(inshape, axis, opset=None, dynamic=False):
        opname = "Softmax"
        outshape = inshape
        node_list = []
        input_node_list = [helper.make_tensor_value_info("in", TensorProto.FLOAT, list(inshape))]
        output_node_list = [helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outshape))]
        input_list = [np.random.uniform(size=inshape).astype(np.float32)]
        softmax_inputs = ["in"]

        if dynamic:
            input_node_list.append(
                helper.make_tensor_value_info("shape", TensorProto.INT64, [len(inshape)])
            )
            input_list.append(np.asarray(inshape))
            reshape_node = helper.make_node("Reshape", ["in", "shape"], ["dynamic_in"])
            softmax_inputs[0] = "dynamic_in"
            node_list += [reshape_node]

        y = helper.make_node(opname, softmax_inputs, ["out"])
        if axis is not None:
            axis_attr = helper.make_attribute("axis", axis)
            y.attribute.append(axis_attr)
        node_list.append(y)

        graph = helper.make_graph(
            node_list,
            opname + "_test",
            inputs=input_node_list,
            outputs=output_node_list,
        )

        model = helper.make_model(graph, producer_name=opname + "_test")
        verify_with_ort_with_inputs(
            model, input_list, use_vm=True, opset=opset, target=target, dev=dev
        )

    verify_softmax((1, 10), None)
    verify_softmax((1, 10), 1)
    verify_softmax((1, 2, 3, 10), 0)
    verify_softmax((1, 2, 3, 10), 2)
    verify_softmax((1, 2, 3, 4, 10), 3)
    verify_softmax((1, 2, 3, 4, 10), 4)
    verify_softmax((1, 10), -1, dynamic=True)
    verify_softmax((1, 2, 3, 10), -1, dynamic=True)
    verify_softmax((1, 10), -1, opset=8, dynamic=True)
    verify_softmax((1, 2, 3, 10), -1, opset=8, dynamic=True)


@tvm.testing.parametrize_targets
def test_forward_min(target, dev):
    """test_forward_min"""

    def verify_min(input_dim):
        dtype = "float32"

        a_np1 = np.random.uniform(size=input_dim).astype(dtype)
        a_np2 = np.random.uniform(size=input_dim).astype(dtype)
        a_np3 = np.random.uniform(size=input_dim).astype(dtype)

        min_node = helper.make_node("Min", ["a_np1", "a_np2", "a_np3"], ["out"])

        graph = helper.make_graph(
            [min_node],
            "Min_test",
            inputs=[
                helper.make_tensor_value_info("a_np1", TensorProto.FLOAT, list(input_dim)),
                helper.make_tensor_value_info("a_np2", TensorProto.FLOAT, list(input_dim)),
                helper.make_tensor_value_info("a_np3", TensorProto.FLOAT, list(input_dim)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(input_dim))],
        )

        model = helper.make_model(graph, producer_name="Min_test")
        verify_with_ort_with_inputs(model, [a_np1, a_np2, a_np3], target=target, dev=dev)

    verify_min((1, 3, 20, 20))
    verify_min((20, 20))


@tvm.testing.parametrize_targets
def test_forward_max(target, dev):
    """test_forward_max"""

    def verify_max(input_dim):
        dtype = "float32"

        a_np1 = np.random.uniform(size=input_dim).astype(dtype)
        a_np2 = np.random.uniform(size=input_dim).astype(dtype)
        a_np3 = np.random.uniform(size=input_dim).astype(dtype)

        max_node = helper.make_node("Max", ["a_np1", "a_np2", "a_np3"], ["out"])

        graph = helper.make_graph(
            [max_node],
            "Max_test",
            inputs=[
                helper.make_tensor_value_info("a_np1", TensorProto.FLOAT, list(input_dim)),
                helper.make_tensor_value_info("a_np2", TensorProto.FLOAT, list(input_dim)),
                helper.make_tensor_value_info("a_np3", TensorProto.FLOAT, list(input_dim)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(input_dim))],
        )

        model = helper.make_model(graph, producer_name="Max_test")
        verify_with_ort_with_inputs(model, [a_np1, a_np2, a_np3], target=target, dev=dev)

    verify_max((1, 3, 20, 20))
    verify_max((20, 20))


@tvm.testing.parametrize_targets
def test_forward_mean(target, dev):
    """test_forward_mean"""

    def verify_mean(input_dim):
        dtype = "float32"

        a_np1 = np.random.uniform(size=input_dim).astype(dtype)
        a_np2 = np.random.uniform(size=input_dim).astype(dtype)
        a_np3 = np.random.uniform(size=input_dim).astype(dtype)

        mean_node = helper.make_node("Mean", ["a_np1", "a_np2", "a_np3"], ["out"])

        graph = helper.make_graph(
            [mean_node],
            "Mean_test",
            inputs=[
                helper.make_tensor_value_info("a_np1", TensorProto.FLOAT, list(input_dim)),
                helper.make_tensor_value_info("a_np2", TensorProto.FLOAT, list(input_dim)),
                helper.make_tensor_value_info("a_np3", TensorProto.FLOAT, list(input_dim)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(input_dim))],
        )

        model = helper.make_model(graph, producer_name="Mean_test")
        verify_with_ort_with_inputs(model, [a_np1, a_np2, a_np3], target=target, dev=dev)

    verify_mean((1, 3, 20, 20))
    verify_mean((20, 20))


@tvm.testing.parametrize_targets
def test_forward_hardsigmoid(target, dev):
    """test_forward_hardsigmoid"""

    def verify_hardsigmoid(input_dim, alpha, beta):
        dtype = "float32"

        a_np1 = np.random.uniform(size=input_dim).astype(dtype)

        hardsigmoid_node = helper.make_node(
            "HardSigmoid", ["a_np1"], ["out"], alpha=alpha, beta=beta
        )

        graph = helper.make_graph(
            [hardsigmoid_node],
            "HardSigmoid_test",
            inputs=[helper.make_tensor_value_info("a_np1", TensorProto.FLOAT, list(input_dim))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(input_dim))],
        )

        model = helper.make_model(graph, producer_name="HardSigmoid_test")
        verify_with_ort_with_inputs(model, [a_np1], target=target, dev=dev)

    verify_hardsigmoid((1, 3, 20, 20), 0.5, 0.6)
    verify_hardsigmoid((20, 20), 0.3, 0.4)


# TODO (mbrookhart, electriclilies) Fix argmin on GPU and enable this test
@tvm.testing.known_failing_targets("cuda")
@tvm.testing.parametrize_targets
def test_forward_arg_min_max(target, dev):
    """test_forward_arg_min_max"""

    def verify_argreduce(input_dim, op_name, axis=None, keepdims=None):
        a_np1 = np.random.uniform(-10, 10, input_dim).astype(np.int32)
        out_shape = list(a_np1.shape)
        def_axis = axis if axis is not None else 0
        if keepdims == 1 or keepdims is None:
            out_shape[def_axis] = 1
        else:
            out_shape.pop(def_axis)

        node = onnx.helper.make_node(op_name, inputs=["a_np1"], outputs=["out"])

        if keepdims is not None:
            keepdims_attr = helper.make_attribute("keepdims", keepdims)
            node.attribute.append(keepdims_attr)
        if axis is not None:
            axis_attr = helper.make_attribute("axis", axis)
            node.attribute.append(axis_attr)

        graph = helper.make_graph(
            [node],
            "argreduce_test",
            inputs=[helper.make_tensor_value_info("a_np1", TensorProto.INT32, list(a_np1.shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.INT64, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="argreduce_test")
        verify_with_ort_with_inputs(model, [a_np1], target=target, dev=dev)

    # Verify argmin and argmax
    verify_argreduce([3, 4, 4], "ArgMin")
    verify_argreduce([3, 4, 4], "ArgMax")
    verify_argreduce([3, 4, 4], "ArgMin", axis=1)
    verify_argreduce([3, 4, 4], "ArgMax", axis=0)
    verify_argreduce([3, 4, 4], "ArgMin", keepdims=0)
    verify_argreduce([3, 4, 4], "ArgMax", keepdims=1)
    for axis in [None, 0, 1, 2]:
        for keepdims in [None, True, False]:
            verify_argreduce([3, 4, 4], "ArgMin", axis, keepdims)
            verify_argreduce([3, 4, 4], "ArgMax", axis, keepdims)


@tvm.testing.parametrize_targets
def test_constantofshape(target, dev):
    """test_constantofshape"""

    def verify_constantofshape(input_dim, value, dtype):
        fill_node = helper.make_node(
            "ConstantOfShape",
            ["input"],
            ["output"],
            value=helper.make_tensor(
                "value", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], (1,), (value,)
            ),
        )

        inputs = [helper.make_tensor_value_info("input", TensorProto.INT64, [len(input_dim)])]

        graph = helper.make_graph(
            [fill_node],
            "fill_test",
            inputs,
            outputs=[
                helper.make_tensor_value_info(
                    "output", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], input_dim
                )
            ],
        )

        model = helper.make_model(graph, producer_name="fill_test")
        input_np = np.array(input_dim).astype("int64")
        verify_with_ort_with_inputs(model, [input_np], use_vm=True, target=target, dev=dev)

    verify_constantofshape((2, 3, 4, 5), 10, "float32")
    verify_constantofshape((3, 3), 0, "int32")
    verify_constantofshape((1, 2, 3), -1, "float32")


@tvm.testing.parametrize_targets
def test_pad(target, dev):
    """test_pad"""

    def verify_pad(indata, pads, mode="constant", value=0.0):
        indata = np.array(indata).astype(np.float32)
        #  numpy expect result
        len_dim = len(pads) // 2
        np_pads = [(pads[i], pads[i + len_dim]) for i in range(len_dim)]
        #  onnx graph
        if mode in ["edge", "reflect"]:
            outdata = np.pad(indata, pad_width=np_pads, mode=mode)
            node = helper.make_node(
                "Pad",
                inputs=["input"],
                outputs=["output"],
                mode=mode,
                pads=pads,
            )
        else:
            outdata = np.pad(indata, pad_width=np_pads, mode="constant", constant_values=value)
            node = helper.make_node(
                "Pad", inputs=["input"], outputs=["output"], mode="constant", pads=pads, value=value
            )
        graph = helper.make_graph(
            [node],
            "pad_test",
            inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape))],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
            ],
        )
        model = helper.make_model(graph, producer_name="pad_test")
        verify_with_ort_with_inputs(
            model, [indata], [outdata.shape], dtype="float32", opset=2, target=target, dev=dev
        )

    def verify_pad_v11(indata, pads, mode="constant", value=0.0):
        indata = np.array(indata).astype(np.float32)
        #  numpy expect result
        len_dim = len(pads) // 2
        np_pads = [(pads[i], pads[i + len_dim]) for i in range(len_dim)]
        pads = np.array(pads)
        #  onnx graph
        if mode in ["edge", "reflect"]:
            inputs = [indata]
            outdata = np.pad(indata, pad_width=np_pads, mode=mode)
            node = helper.make_node("Pad", inputs=["input", "pads"], outputs=["output"], mode=mode)
            graph = helper.make_graph(
                [node],
                "pad_test",
                inputs=[
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape)),
                    helper.make_tensor_value_info("pads", TensorProto.INT64, (len(pads),)),
                ],
                initializer=[helper.make_tensor("pads", TensorProto.INT64, (len(pads),), pads)],
                outputs=[
                    helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
                ],
            )
        else:
            inputs = [indata]
            outdata = np.pad(indata, pad_width=np_pads, mode="constant", constant_values=value)
            node = helper.make_node(
                "Pad",
                inputs=["input", "pads", "constant_value"],
                outputs=["output"],
                mode="constant",
            )
            graph = helper.make_graph(
                [node],
                "pad_test",
                inputs=[
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape)),
                    helper.make_tensor_value_info("pads", TensorProto.INT64, (len(pads),)),
                    helper.make_tensor_value_info("constant_value", TensorProto.FLOAT, (1,)),
                ],
                initializer=[
                    helper.make_tensor("pads", TensorProto.INT64, (len(pads),), pads),
                    helper.make_tensor("constant_value", TensorProto.FLOAT, (1,), [value]),
                ],
                outputs=[
                    helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
                ],
            )
        model = helper.make_model(graph, producer_name="pad_test")
        verify_with_ort_with_inputs(model, inputs, opset=11, use_vm=True, target=target, dev=dev)

    verify_pad(np.random.randn(2, 2).astype(np.float32), [0, 1, 0, 0], "constant", 0.0)
    verify_pad(np.random.randn(2, 3).astype(np.float32), [1, 0, 0, 1], "constant", 0.0)
    verify_pad(np.random.randn(3, 2).astype(np.float32), [0, 0, 1, 0], "constant", 5.0)
    verify_pad(np.random.randn(1, 3, 4, 5).astype(np.float32), [0, 0, 1, 1, 0, 0, 1, 1], "edge")
    verify_pad(np.random.randn(1, 3, 4, 5).astype(np.float32), [0, 0, 1, 1, 0, 0, 1, 1], "reflect")

    verify_pad_v11(np.random.randn(2, 2).astype(np.float32), [0, 1, 0, 0], "constant", 0.0)
    verify_pad_v11(np.random.randn(2, 3).astype(np.float32), [1, 0, 0, 1], "constant", 0.0)
    verify_pad_v11(np.random.randn(3, 2).astype(np.float32), [0, 0, 1, 0], "constant", 5.0)
    verify_pad_v11(np.random.randn(1, 3, 4, 5).astype(np.float32), [0, 0, 1, 1, 0, 0, 1, 1], "edge")
    verify_pad_v11(
        np.random.randn(1, 3, 4, 5).astype(np.float32), [0, 0, 1, 1, 0, 0, 1, 1], "reflect"
    )


@tvm.testing.parametrize_targets
def test_all_reduce_funcs(target, dev):
    """test_all_reduce_funcs"""

    def verify_reduce_func(func, data, axis, keepdims):
        inshape = data.shape
        outshape = np.sum(data, axis=axis, keepdims=keepdims == 1).shape

        if axis:
            node = onnx.helper.make_node(
                func, inputs=["x"], outputs=["y"], axes=axis, keepdims=keepdims
            )
        else:
            node = onnx.helper.make_node(func, inputs=["x"], outputs=["y"], keepdims=keepdims)

        graph = helper.make_graph(
            [node],
            "reduce_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(inshape))],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(outshape))],
        )

        model = helper.make_model(graph, producer_name="reduce_test")

        verify_with_ort_with_inputs(
            model,
            [data],
            [outshape],
            opset=11,
            target=target,
            dev=dev,
            rtol=1e-4,
            atol=1e-4,
        )

    funcs = [
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "ReduceSumSquare",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceL1",
        "ReduceL2",
    ]

    for func in funcs:
        verify_reduce_func(func, np.array(1.0).astype(np.float32), axis=None, keepdims=False)

        for keepdims in [True, False]:
            verify_reduce_func(
                func, np.random.randn(3, 2, 2).astype(np.float32), axis=None, keepdims=keepdims
            )

            verify_reduce_func(
                func, np.random.randn(3, 2, 3).astype(np.float32), axis=None, keepdims=keepdims
            )

            verify_reduce_func(
                func, np.random.randn(3, 3, 3).astype(np.float32), axis=(1,), keepdims=keepdims
            )

            verify_reduce_func(
                func, np.random.randn(3, 3, 3, 1).astype(np.float32), axis=(1, 2), keepdims=keepdims
            )

            verify_reduce_func(
                func, np.random.randn(3, 3, 3, 1).astype(np.float32), axis=(1,), keepdims=keepdims
            )

            verify_reduce_func(
                func, np.random.randn(1, 3, 4, 1).astype(np.float32), axis=(1,), keepdims=keepdims
            )


@tvm.testing.parametrize_targets
def test_split(target, dev):
    """test_split"""

    def verify_split(indata, outdatas, split, axis=0, pass_split=True, opset=11):
        indata = np.array(indata).astype(np.float32)
        outdatas = [np.array(o).astype(np.float32) for o in outdatas]
        inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape))]
        input_names = ["input"]
        initializer = []

        if split:
            split_index = range(len(split))
        else:
            split_index = range(len(outdatas))

        if pass_split:
            if opset >= 13:
                input_names.append("split")
                np_split = np.array(split).astype(np.int64)
                inputs.append(
                    helper.make_tensor_value_info("split", TensorProto.INT64, list(np_split.shape))
                )
                # TODO(mbrookhart): Support dynamic split, edit this test case to remove split from
                # the initializer and add it back to the input data
                indata = [indata]  # , np_split]
                initializer.append(
                    helper.make_tensor("split", TensorProto.INT64, list(np_split.shape), np_split)
                )
        node = helper.make_node(
            "Split",
            inputs=input_names,
            outputs=[f"output_{i}" for i in range(len(split_index))],
            axis=axis,
        )

        if pass_split and opset < 13:
            split_attr = helper.make_attribute("split", split)
            node.attribute.append(split_attr)

        graph = helper.make_graph(
            [node],
            "split_test",
            inputs=inputs,
            initializer=initializer,
            outputs=[
                helper.make_tensor_value_info(
                    f"output_{i}", TensorProto.FLOAT, list(outdatas[i].shape)
                )
                for i in range(len(split_index))
            ],
        )
        model = helper.make_model(graph, producer_name="split_test")
        verify_with_ort_with_inputs(
            model,
            indata,
            out_shape=list(range(len(split_index))),
            opset=opset,
            target=target,
            dev=dev,
            use_vm=True,
            freeze_params=(opset >= 13),
        )

    # 1D
    verify_split([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [2, 2, 2], 0)
    verify_split(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [2, 2, 2], 0, False
    )
    verify_split([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]], [2, 1, 3], 0)
    verify_split(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]], [2, 1, 3], 0, opset=13
    )
    # 2D
    verify_split(
        [[1.0, 2.0, 3.0, 4.0], [7.0, 8.0, 9.0, 10.0]],
        [[[1.0, 2.0], [7.0, 8.0]], [[3.0, 4.0], [9.0, 10.0]]],
        [2, 2],
        1,
    )
    verify_split(
        [[1.0, 2.0, 3.0, 4.0], [7.0, 8.0, 9.0, 10.0]],
        [[[1.0, 2.0], [7.0, 8.0]], [[3.0, 4.0], [9.0, 10.0]]],
        [2, 2],
        1,
        opset=13,
    )
    # Split evenly (unstack)
    verify_split([1, 2, 3], [[1], [2], [3]], False, 0, False)
    # Split a single value to a single value
    verify_split([1], [[1]], [1], pass_split=True)
    # Test that the default case modifies nothing when split list has length one
    verify_split([[1.0, 2.0]], [[1.0, 2.0]], [2], 1)
    verify_split([[1.0, 2.0]], [[1.0, 2.0]], [1], 0)


@tvm.testing.parametrize_targets
def test_binary_ops(target, dev):
    """test_binary_ops"""
    in_shape = (1, 2, 3, 3)
    dtype = "float32"
    out_shape = in_shape

    def verify_binary_ops(op, x, y, out_type="float32"):
        out = helper.make_node(op, ["in1", "in2"], ["out"])
        graph = helper.make_graph(
            [out],
            "_test",
            inputs=[
                helper.make_tensor_value_info("in1", TensorProto.FLOAT, x.shape),
                helper.make_tensor_value_info("in2", TensorProto.FLOAT, y.shape),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "out", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(out_type)], list(out_shape)
                )
            ],
        )
        model = helper.make_model(graph, producer_name="_test")
        verify_with_ort_with_inputs(model, [x, y], target=target, dev=dev)

    x = np.random.uniform(size=in_shape).astype(dtype)
    y = np.random.uniform(size=in_shape).astype(dtype)
    z_array = np.random.uniform(size=(3,)).astype(dtype)
    verify_binary_ops("Add", x, y)
    verify_binary_ops("Add", x, z_array)
    verify_binary_ops("Sub", x, y)
    verify_binary_ops("Sub", x, z_array)
    verify_binary_ops("Mul", x, y)
    verify_binary_ops("Mul", x, z_array)
    verify_binary_ops("Div", x, y)
    verify_binary_ops("Div", x, z_array)
    verify_binary_ops("Sum", x, y)
    verify_binary_ops("Sum", x, z_array)
    verify_binary_ops("Greater", x, y, "bool")
    verify_binary_ops("Greater", x, z_array, "bool")
    verify_binary_ops("GreaterOrEqual", x, y, "bool")
    verify_binary_ops("GreaterOrEqual", x, z_array, "bool")
    verify_binary_ops("Less", x, y, "bool")
    verify_binary_ops("Less", x, z_array, "bool")
    verify_binary_ops("LessOrEqual", x, y, "bool")
    verify_binary_ops("LessOrEqual", x, z_array, "bool")
    verify_binary_ops("Equal", x, y, "bool")
    verify_binary_ops("Equal", x, z_array, "bool")


@tvm.testing.parametrize_targets
def test_unary_ops(target, dev):
    """test_unary_ops"""
    in_shape = (1, 2, 3, 3)
    _ = "float32"
    out_shape = in_shape

    def verify_unary_ops(op, x, rtol=1e-5, atol=1e-5, dtype="float32"):
        x = x.astype(dtype)
        ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
        out = helper.make_node(op, ["in1"], ["out"])
        graph = helper.make_graph(
            [out],
            "_test",
            inputs=[
                helper.make_tensor_value_info("in1", ONNX_DTYPE, list(in_shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", ONNX_DTYPE, list(out_shape))],
        )
        model = helper.make_model(graph, producer_name="_test")
        verify_with_ort_with_inputs(model, [x], rtol=rtol, atol=atol, target=target, dev=dev)

    x = np.random.uniform(size=in_shape)
    verify_unary_ops("Neg", x)
    verify_unary_ops("Abs", x)
    verify_unary_ops("Reciprocal", x)
    verify_unary_ops("Reciprocal", x, dtype="float16")
    verify_unary_ops("Sqrt", x)
    verify_unary_ops("Relu", x)
    verify_unary_ops("Exp", x)
    verify_unary_ops("Log", x)
    verify_unary_ops("Log", x)
    verify_unary_ops("Acos", x)
    verify_unary_ops("Acosh", x)
    verify_unary_ops("Asin", x)
    verify_unary_ops("Asinh", x)
    verify_unary_ops("Atan", x)
    verify_unary_ops("Atanh", x)
    verify_unary_ops("Cos", x)
    verify_unary_ops("Cosh", x)
    verify_unary_ops("Sin", x)
    verify_unary_ops("Sinh", x)
    verify_unary_ops("Tan", x)
    verify_unary_ops("Tanh", x)
    verify_unary_ops("Sigmoid", x)
    verify_unary_ops("Softsign", x)


@tvm.testing.parametrize_targets
def test_leaky_relu(target, dev):
    def leaky_relu_x(x, alpha):
        return np.where(x >= 0, x, x * alpha)

    _test_onnx_op_elementwise(
        target,
        dev,
        (2, 4, 5, 6),
        leaky_relu_x,
        {"alpha": 0.25},
        "float32",
        "LeakyRelu",
        {"alpha": 0.25},
    )


@tvm.testing.parametrize_targets
def test_elu(target, dev):
    def elu_x(x, alpha):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1.0))

    _test_onnx_op_elementwise(
        target, dev, (2, 4, 5, 6), elu_x, {"alpha": 0.25}, "float32", "Elu", {"alpha": 0.25}
    )


@tvm.testing.parametrize_targets
def test_selu(target, dev):
    def selu_x(x, alpha, gamma):
        return gamma * np.where(x > 0, x, alpha * (np.exp(x) - 1.0))

    _test_onnx_op_elementwise(
        target,
        dev,
        (2, 4, 5, 6),
        selu_x,
        {"alpha": 0.25, "gamma": 0.3},
        "float32",
        "Selu",
        {"alpha": 0.25, "gamma": 0.3},
    )


@tvm.testing.parametrize_targets
def test_prelu(target, dev):
    """test_prelu"""

    def verify_prelu(x_shape, a_shape):
        node = helper.make_node("PRelu", inputs=["X", "slope"], outputs=["Y"])

        graph = helper.make_graph(
            [node],
            "prelu_test",
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x_shape)),
                helper.make_tensor_value_info("slope", TensorProto.FLOAT, list(a_shape)),
            ],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(x_shape))],
        )

        model = helper.make_model(graph, producer_name="prelu_test")

        verify_with_ort(
            model,
            [x_shape, a_shape],
            out_shape=[list(x_shape)],
            use_vm=True,
            target=target,
            dev=dev,
        )

    verify_prelu([3, 4, 5, 6], [1, 4, 1, 1])
    verify_prelu([1, 8, 5, 6], [1, 8, 1, 1])
    verify_prelu([2, 12, 16, 16], [1, 12, 1, 1])
    verify_prelu([2, 12, 16, 16], [1])  # Test alpha broadcasting.
    verify_prelu([3, 1], [3, 1])  # Test non NCHW workload.


@tvm.testing.parametrize_targets
def test_thresholded_relu(target, dev):
    def thresholded_relu_x(x, alpha):
        out_np = np.clip(x, alpha, np.inf)
        out_np[out_np == alpha] = 0
        return out_np

    _test_onnx_op_elementwise(
        target,
        dev,
        (2, 4, 5, 6),
        thresholded_relu_x,
        {"alpha": 0.25},
        "float32",
        "ThresholdedRelu",
        {"alpha": 0.25},
    )


@tvm.testing.parametrize_targets
def test_logsoftmax(target, dev):
    _test_onnx_op_elementwise(
        target,
        dev,
        (1, 4),
        tvm.topi.testing.log_softmax_python,
        {},
        "float32",
        "LogSoftmax",
        {"axis": 1},
    )