@tvm.testing.parametrize_targets
def test_sign(target, dev):
    def sign_x(x):
        return np.sign(x)

    _test_onnx_op_elementwise(target, dev, (3, 4, 5, 6), sign_x, {}, "float32", "Sign", {})


@tvm.testing.parametrize_targets
def test_not(target, dev):
    """test_not"""

    def verify_not(indata, dtype):
        x = indata.astype(dtype)

        node = helper.make_node(
            "Not",
            inputs=["in"],
            outputs=["out"],
        )

        graph = helper.make_graph(
            [node],
            "not_test",
            inputs=[helper.make_tensor_value_info("in", TensorProto.BOOL, list(x.shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.BOOL, list(x.shape))],
        )

        model = helper.make_model(graph, producer_name="not_test")
        verify_with_ort_with_inputs(model, [x], target=target, dev=dev)

    # 2d
    verify_not(indata=(np.random.randn(3, 4) > 0), dtype=bool)
    # 3d
    verify_not(indata=(np.random.randn(3, 4, 5) > 0), dtype=bool)
    # 4d
    verify_not(indata=(np.random.randn(3, 4, 5, 6) > 0), dtype=bool)


@tvm.testing.parametrize_targets
def test_and(target, dev):
    """test_and"""

    def verify_and(indata, dtype):
        x = indata[0].astype(dtype)
        y = indata[1].astype(dtype)
        outdata = np.logical_and(x, y)

        node = helper.make_node(
            "And",
            inputs=["in1", "in2"],
            outputs=["out"],
        )

        graph = helper.make_graph(
            [node],
            "and_test",
            inputs=[
                helper.make_tensor_value_info("in1", TensorProto.BOOL, list(x.shape)),
                helper.make_tensor_value_info("in2", TensorProto.BOOL, list(y.shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.BOOL, list(outdata.shape))],
        )

        model = helper.make_model(graph, producer_name="and_test")
        verify_with_ort_with_inputs(model, [x, y], [outdata.shape], target=target, dev=dev)

    # 2d
    x = np.random.randn(3, 4) > 0
    y = np.random.randn(3, 4) > 0
    verify_and(indata=[x, y], dtype=bool)

    # 3d
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(3, 4, 5) > 0
    verify_and(indata=[x, y], dtype=bool)

    # 4d
    x = np.random.randn(3, 4, 5, 6) > 0
    y = np.random.randn(3, 4, 5, 6) > 0
    verify_and(indata=[x, y], dtype=bool)

    # 3d vs 1d
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(5) > 0
    verify_and(indata=[x, y], dtype=bool)

    # 3d vs 2d
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(4, 5) > 0
    verify_and(indata=[x, y], dtype=bool)


@tvm.testing.parametrize_targets
def test_tile(target, dev):
    """test_tile"""

    def verify_tile_v6(indata, repeats, outdata):
        node = helper.make_node("Tile", inputs=["input", "repeats"], outputs=["out"])
        graph = helper.make_graph(
            [node],
            "tile_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape)),
                helper.make_tensor_value_info("repeats", TensorProto.INT64, list(repeats.shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outdata.shape))],
        )

        model = helper.make_model(graph, producer_name="tile_test")
        verify_with_ort_with_inputs(
            model, [indata, repeats], use_vm=True, opset=6, target=target, dev=dev
        )

    x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    repeats = np.random.randint(low=1, high=10, size=(np.ndim(x),)).astype(np.int64)
    z_array = np.tile(x, repeats)
    verify_tile_v6(x, repeats, z_array)


@tvm.testing.parametrize_targets
def test_erf(target, dev):
    """test_erf"""

    def verify_erf(indata, outdata):
        node = helper.make_node("Erf", inputs=["in"], outputs=["out"])
        graph = helper.make_graph(
            [node],
            "erf_test",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(indata.shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outdata.shape))],
        )
        model = helper.make_model(graph, producer_name="erf_test")
        verify_with_ort_with_inputs(model, [indata], [outdata.shape], target=target, dev=dev)

    x = np.random.rand(2, 3, 4, 6).astype(np.float32)
    z_array = scipy.special.erf(x)
    verify_erf(x, z_array)


@tvm.testing.parametrize_targets
def test_where(target, dev):
    """test_where"""

    def verify_where(condition, x, y, dtype, outdata, dynamic=False):
        node_list = []
        where_inputs = ["condition", "x", "y"]
        if dynamic:
            shape_node = helper.make_node("Shape", ["x"], ["shape"])
            reshape_node = helper.make_node("Reshape", ["x", "shape"], ["X"])
            where_inputs[1] = "X"
            node_list += [shape_node, reshape_node]
        node = helper.make_node("Where", inputs=where_inputs, outputs=["out"])
        node_list.append(node)
        graph = helper.make_graph(
            node_list,
            "where_test",
            inputs=[
                helper.make_tensor_value_info("condition", TensorProto.BOOL, list(condition.shape)),
                helper.make_tensor_value_info("x", dtype, list(x.shape)),
                helper.make_tensor_value_info("y", dtype, list(y.shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", dtype, list(outdata.shape))],
        )
        model = helper.make_model(graph, producer_name="where_test")
        verify_with_ort_with_inputs(
            model, [condition, x, y], [outdata.shape], use_vm=True, target=target, dev=dev
        )

    condition = np.array([[1, 0], [1, 1]], dtype=bool)
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([[9, 8], [7, 6]], dtype=np.int64)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.INT64, outdata)

    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[9, 8], [7, 6]], dtype=np.float32)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata)

    x = np.array(1, dtype=np.float32)
    y = np.array([2], dtype=np.float32)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata)

    x = np.array([2], dtype=np.float32)
    y = np.array(1, dtype=np.float32)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata)

    condition = np.array(1, dtype=bool)
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[5, 6], [7, 8]], dtype=np.float32)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata)

    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[1], [7]], dtype=np.float32)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata, dynamic=True)

    condition = np.random.uniform(size=(3, 1)) < 0.5
    x = np.random.uniform(size=2).astype(np.float32)
    y = np.random.uniform(size=2).astype(np.float32)
    outdata = np.where(condition, x, y)
    verify_where(condition, x, y, TensorProto.FLOAT, outdata)


@tvm.testing.parametrize_targets
def test_or(target, dev):
    """test_or"""

    def verify_or(indata, dtype):
        x = indata[0].astype(dtype)
        y = indata[1].astype(dtype)
        outdata = np.logical_or(x, y)

        node = helper.make_node(
            "Or",
            inputs=["in1", "in2"],
            outputs=["out"],
        )

        graph = helper.make_graph(
            [node],
            "or_test",
            inputs=[
                helper.make_tensor_value_info("in1", TensorProto.BOOL, list(x.shape)),
                helper.make_tensor_value_info("in2", TensorProto.BOOL, list(y.shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.BOOL, list(outdata.shape))],
        )

        model = helper.make_model(graph, producer_name="or_test")
        verify_with_ort_with_inputs(model, [x, y], [outdata.shape], target=target, dev=dev)

    # 2d
    x = np.random.randn(3, 4) > 0
    y = np.random.randn(3, 4) > 0
    verify_or(indata=[x, y], dtype=bool)

    # 3d
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(3, 4, 5) > 0
    verify_or(indata=[x, y], dtype=bool)

    # 4d
    x = np.random.randn(3, 4, 5, 6) > 0
    y = np.random.randn(3, 4, 5, 6) > 0
    verify_or(indata=[x, y], dtype=bool)

    # 3d vs 1d
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(5) > 0
    verify_or(indata=[x, y], dtype=bool)

    # 3d vs 2d
    x = np.random.randn(3, 4, 5) > 0
    y = np.random.randn(4, 5) > 0
    verify_or(indata=[x, y], dtype=bool)


@tvm.testing.parametrize_targets
def test_batch_norm(target, dev):
    """test_batch_norm"""

    def verify_batch_norm(in_shape):
        batchnorm = onnx.helper.make_node(
            "BatchNormalization", inputs=["x", "scale", "B", "mean", "var"], outputs=["Y"]
        )

        graph = helper.make_graph(
            [batchnorm],
            "batchnorm_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(in_shape)),
                helper.make_tensor_value_info("scale", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("mean", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("var", TensorProto.FLOAT, [in_shape[1]]),
            ],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(in_shape))],
        )

        model = helper.make_model(graph, producer_name="batchnorm_test")
        # X, scale, b, mean, var
        inshapes = [in_shape, in_shape[1], in_shape[1], in_shape[1], in_shape[1]]
        verify_with_ort(model, inshapes, out_shape=[in_shape], target=target, dev=dev)

    verify_batch_norm([1, 3, 224, 224])
    verify_batch_norm([1, 3, 24, 24])
    verify_batch_norm([16, 3, 24, 24])
    verify_batch_norm([16, 16, 24, 24])
    verify_batch_norm([16, 16, 10, 10])


@tvm.testing.parametrize_targets
def test_batch_norm_dynamic_subgraph(target, dev):
    """test_batch_norm_dynamic_subgraph"""

    def verify_batch_norm_dynamic_subgraph(in_shape, o_shape):

        batchnorm = onnx.helper.make_node(
            "BatchNormalization", inputs=["x", "scale", "B", "mean", "var"], outputs=["Y"]
        )

        shape_node = helper.make_node("Shape", ["Y"], ["shape"])
        reshape_node = helper.make_node("Reshape", ["in", "shape"], ["out"])
        graph = helper.make_graph(
            [batchnorm, shape_node, reshape_node],
            "batchnorm_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(in_shape)),
                helper.make_tensor_value_info("in", TensorProto.FLOAT, list(o_shape)),
                helper.make_tensor_value_info("scale", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("mean", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("var", TensorProto.FLOAT, [in_shape[1]]),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(in_shape))],
        )

        model = helper.make_model(graph, producer_name="batchnorm_test")

        # X, inp, scale, b, mean, var
        inshapes = [in_shape, o_shape, in_shape[1], in_shape[1], in_shape[1], in_shape[1]]
        verify_with_ort(model, inshapes, out_shape=[in_shape], use_vm=True, target=target, dev=dev)

    verify_batch_norm_dynamic_subgraph([16, 16, 10, 10], [160, 160])


@tvm.testing.parametrize_targets
def test_conv(target, dev):
    """test_conv"""

    def verify_conv(
        x_shape,
        w_shape,
        y_shape,
        padding,
        kernel_shape,
        strides,
        dilations,
        group=1,
        auto_pad="NOTSET",
        unset_pad=False,
    ):
        if unset_pad:
            node = helper.make_node(
                "Conv",
                inputs=["x", "W"],
                outputs=["y"],
                kernel_shape=kernel_shape,
                # Default values for other attributes:
                strides=strides,
                dilations=dilations,
                group=group,
            )
        elif padding is None:
            ## autopadding with unset default attributes
            kwargs = {}
            if not all(list(s == 1 for s in strides)):
                kwargs["strides"] = strides
            if not all(list(d == 1 for d in dilations)):
                kwargs["dilations"] = dilations

            node = helper.make_node(
                "Conv",
                inputs=["x", "W"],
                outputs=["y"],
                # Default values for other attributes:
                auto_pad=auto_pad,
                group=group,
                **kwargs,
            )
        else:
            node = helper.make_node(
                "Conv",
                inputs=["x", "W"],
                outputs=["y"],
                kernel_shape=kernel_shape,
                # Default values for other attributes:
                strides=strides,
                dilations=dilations,
                group=group,
                pads=padding,
            )

        graph = helper.make_graph(
            [node],
            "conv_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape)),
                helper.make_tensor_value_info("W", TensorProto.FLOAT, list(w_shape)),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))],
        )

        model = helper.make_model(graph, producer_name="conv_test")

        verify_with_ort(
            model,
            [x_shape, w_shape],
            [y_shape],
            use_vm=True,
            target=target,
            dev=dev,
        )

    def repeat(num, dims):
        return tuple(num for _ in range(dims))

    for dims in [1, 2, 3]:
        # Convolution with padding
        verify_conv(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            (1, 1) + repeat(5, dims),
            2 * repeat(1, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
        )
        # Convolution with asymmetric padding
        verify_conv(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            (1, 1) + repeat(4, dims),
            repeat(0, dims) + repeat(1, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
        )
        # Convolution without padding
        verify_conv(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            (1, 1) + repeat(3, dims),
            2 * repeat(0, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
        )
        # Convolution with autopadding
        verify_conv(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            (1, 1) + repeat(5, dims),
            None,
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
            auto_pad="SAME_UPPER",
        )
        # Convolution with valid autopadding
        verify_conv(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            (1, 1) + repeat(3, dims),
            None,
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
            auto_pad="VALID",
        )
        # Convolution with unset padding
        verify_conv(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            (1, 1) + repeat(3, dims),
            2 * repeat(0, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
            True,
        )
        # Convolution with non uniform stride
        verify_conv(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            (1, 1) + repeat(3, dims),
            None,
            repeat(3, dims),
            repeat(2, dims),
            repeat(1, dims),
            auto_pad="SAME_UPPER",
        )
        # Convolution with dilation
        verify_conv(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            (1, 1) + repeat(5, dims),
            2 * repeat(2, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(2, dims),
        )

    # TODO(jwfromm): Merge with other tests once group_conv3d is supported.
    for dims in [1, 2, 3]:
        # Group Convolution
        verify_conv(
            (1, 8) + repeat(5, dims),
            (8, 1) + repeat(3, dims),
            (1, 8) + repeat(5, dims),
            2 * repeat(1, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
            group=8,
        )

        verify_conv(
            (1, 12) + repeat(5, dims),
            (30, 4) + repeat(3, dims),
            (1, 30) + repeat(5, dims),
            2 * repeat(1, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
            group=3,
        )


@tvm.testing.parametrize_targets
def test_convtranspose(target, dev):
    """test_convtranspose"""

    def verify_convtranspose_with_output_shape(
        x_shape,
        w_shape,
        output_shape,
        kernel_shape,
        strides,
        dilations,
        auto_pad="SAME_UPPER",
        group=1,
    ):
        node = helper.make_node(
            "ConvTranspose",
            inputs=["x", "W"],
            outputs=["y"],
            kernel_shape=kernel_shape,
            # Default values for other attributes:
            strides=strides,
            dilations=dilations,
            output_shape=output_shape,
            auto_pad=auto_pad,
        )

        if group is not None:
            group_attr = helper.make_attribute("group", group)
            node.attribute.append(group_attr)

        graph = helper.make_graph(
            [node],
            "ConvTranspose_with_output_shape_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape)),
                helper.make_tensor_value_info("W", TensorProto.FLOAT, list(w_shape)),
            ],
            outputs=[
                helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1] + list(output_shape))
            ],
        )

        model = helper.make_model(graph, producer_name="convtranspose_output_shape_test")

        verify_with_ort(model, [x_shape, w_shape], use_vm=True, target=target, dev=dev)

    def verify_convtranspose_with_padding(
        x_shape,
        w_shape,
        padding,
        kernel_shape,
        strides,
        dilations,
        auto_pad="NOTSET",
        unset_pad=False,
        group=1,
    ):
        node = helper.make_node(
            "ConvTranspose",
            inputs=["x", "W"],
            outputs=["y"],
            kernel_shape=kernel_shape,
            # Default values for other attributes:
            strides=strides,
            dilations=dilations,
        )
        if not unset_pad:
            if padding is None:
                pad_attr = helper.make_attribute("auto_pad", auto_pad)
            else:
                pad_attr = helper.make_attribute("pads", padding)
            node.attribute.append(pad_attr)

        if group is not None:
            group_attr = helper.make_attribute("group", group)
            node.attribute.append(group_attr)

        graph = helper.make_graph(
            [node],
            "convtranspose_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape)),
                helper.make_tensor_value_info("W", TensorProto.FLOAT, list(w_shape)),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, ["?"] * len(x_shape))],
        )

        model = helper.make_model(graph, producer_name="convtranspose_pad_test")

        verify_with_ort(model, [x_shape, w_shape], use_vm=True, target=target, dev=dev)

    def verify_convtranspose(x_shape, w_shape, y_shape, p, group=1):
        node = onnx.helper.make_node(
            "ConvTranspose",
            inputs=["x", "W"],
            outputs=["y"],
            strides=[3, 2],
            kernel_shape=[3, 3],
            pads=p,
        )

        if group is not None:
            group_attr = helper.make_attribute("group", group)
            node.attribute.append(group_attr)

        graph = helper.make_graph(
            [node],
            "verify_convtranspose_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape)),
                helper.make_tensor_value_info("W", TensorProto.FLOAT, list(w_shape)),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))],
        )

        model = helper.make_model(graph, producer_name="convtranspose_test")
        verify_with_ort(model, [x_shape, w_shape], y_shape, opset=11, target=target, dev=dev)

    # Convolution Transpose with padding
    # (1, 1, 3, 3) input tensor
    # (1, 2, 3, 3) tensor for convolution weights
    # (1, 2, 7, 3) output tensor
    # [1, 2, 1, 2] list for pads
    verify_convtranspose((1, 1, 3, 3), (1, 2, 3, 3), (1, 2, 7, 3), [1, 2, 1, 2])
    # Test undefined groups.
    verify_convtranspose((1, 1, 3, 3), (1, 2, 3, 3), (1, 2, 7, 3), [1, 2, 1, 2], group=None)

    if "llvm" in target:
        # GPU does not support groups != 1 for convtranspose, so only test llvm
        # Test depthwise-convolution
        verify_convtranspose((1, 10, 3, 3), (10, 1, 3, 3), (1, 10, 7, 3), [1, 2, 1, 2], group=10)

        # Test grouped-convolution
        verify_convtranspose((1, 10, 3, 3), (10, 1, 3, 3), (1, 5, 7, 3), [1, 2, 1, 2], group=5)

    def repeat(num, dims):
        return tuple(num for _ in range(dims))

    # Once onnxruntime update is complete
    for dims in [1, 2, 3]:
        # Convolution with padding
        verify_convtranspose_with_padding(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            2 * repeat(1, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
        )
        # Convolution without padding
        verify_convtranspose_with_padding(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            2 * repeat(0, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
        )
        # Convolution with unset padding
        verify_convtranspose_with_padding(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            2 * repeat(0, dims),
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
            True,
        )
        # Convolution with autopadding
        verify_convtranspose_with_padding(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            None,
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
            auto_pad="SAME_UPPER",
        )
        # Convolution with valid autopadding
        verify_convtranspose_with_padding(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            None,
            repeat(3, dims),
            repeat(1, dims),
            repeat(1, dims),
            auto_pad="VALID",
        )
        # Convolution with non uniform stride
        verify_convtranspose_with_padding(
            (1, 1) + repeat(5, dims),
            (1, 1) + repeat(3, dims),
            None,
            repeat(3, dims),
            repeat(2, dims),
            repeat(1, dims),
            auto_pad="SAME_UPPER",
        )
        # Convolution with dilation
        # TODO(mbrookhart): Relay doesn't currently support convtranspose with dilation
        # verify_convtranspose_with_padding(
        #     (1, 1) + repeat(5, D),
        #     (1, 1) + repeat(3, D),
        #     2 * repeat(2, D),
        #     repeat(3, D),
        #     repeat(1, D),
        #     repeat(2, D),
        # )

    # Convolution with output_shape
    for dims in [1, 2, 3]:
        for num in range(60, 66):
            verify_convtranspose_with_output_shape(
                (1, 1) + repeat(32, dims),
                (1, 1) + repeat(4, dims),
                repeat(num, dims),
                repeat(4, dims),
                repeat(2, dims),
                repeat(1, dims),
            )

            verify_convtranspose_with_output_shape(
                (1, 1) + repeat(32, dims),
                (1, 1) + repeat(4, dims),
                repeat(num, dims),
                repeat(4, dims),
                repeat(2, dims),
                repeat(1, dims),
                auto_pad="SAME_LOWER",
            )



@tvm.testing.parametrize_targets
def test_pooling(target, dev):
    """test_pooling"""

    def verify_pooling(x_shape, kernel_shape, strides, pads, out_shape, mode, auto_pad="NOTSET"):
        _ = np.random.uniform(size=x_shape).astype("float32")

        if mode == "max":
            node_type = "MaxPool"
        elif mode == "average":
            node_type = "AveragePool"
        else:
            raise ValueError(f"Pool method {mode} is not supported.")

        pool_node = helper.make_node(
            node_type, inputs=["x"], outputs=["y"], kernel_shape=kernel_shape, strides=strides
        )

        if pads is None:
            pad_attr = helper.make_attribute("auto_pad", auto_pad)
        else:
            pad_attr = helper.make_attribute("pads", pads)
        pool_node.attribute.append(pad_attr)

        if mode == "max":
            storage_attr = helper.make_attribute("storage_order", 0)
            pool_node.attribute.append(storage_attr)

        graph = helper.make_graph(
            [pool_node],
            "pooling_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="pooling_test")
        verify_with_ort(
            model,
            [x_shape],
            [out_shape],
            use_vm=False,
            target=target,
            dev=dev,
        )

    for mode in ["max", "average"]:
        # Pool1D
        verify_pooling(
            x_shape=[1, 1, 32],
            kernel_shape=[3],
            strides=[1],
            pads=[1, 1],
            out_shape=[1, 1, 32],
            mode=mode,
        )
        # Pool2D
        verify_pooling(
            x_shape=[1, 1, 32, 32],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[1, 1, 1, 1],
            out_shape=[1, 1, 32, 32],
            mode=mode,
        )

        # Pool1D with stride
        verify_pooling(
            x_shape=[1, 1, 32],
            kernel_shape=[3],
            strides=[2],
            pads=[1, 1],
            out_shape=[1, 1, 16],
            mode=mode,
        )
        # Pool2D with stride
        verify_pooling(
            x_shape=[1, 1, 32, 32],
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
            out_shape=[1, 1, 16, 16],
            mode=mode,
        )

        # Pool1D with stride and autopadding
        verify_pooling(
            x_shape=[1, 1, 32],
            kernel_shape=[3],
            strides=[2],
            pads=None,
            out_shape=[1, 1, 16],
            mode=mode,
            auto_pad="SAME_UPPER",
        )
        # Pool2D with stride and autopadding
        verify_pooling(
            x_shape=[1, 1, 32, 32],
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=None,
            out_shape=[1, 1, 16, 16],
            mode=mode,
            auto_pad="SAME_UPPER",
        )

        # Pool3D with stride
        verify_pooling(
            x_shape=[1, 1, 32, 32, 32],
            kernel_shape=[3, 3, 3],
            strides=[2, 2, 2],
            pads=[1, 1, 1, 1, 1, 1],
            out_shape=[1, 1, 16, 16, 16],
            mode=mode,
        )

        # Pool3D with stride and autopadding
        verify_pooling(
            x_shape=[1, 1, 32, 32, 32],
            kernel_shape=[3, 3, 3],
            strides=[2, 2, 2],
            pads=None,
            out_shape=[1, 1, 16, 16, 16],
            mode=mode,
            auto_pad="SAME_UPPER",
        )


@tvm.testing.parametrize_targets
def test_global_pooling(target, dev):
    """test_global_pooling"""

    def verify_global_pooling(x_shape, mode):
        out_shape = x_shape[:2] + [1] * (len(x_shape) - 2)

        if mode == "max":
            node_type = "GlobalMaxPool"
        elif mode == "average":
            node_type = "GlobalAveragePool"
        else:
            raise ValueError(f"Pool method {mode} is not supported.")

        pool_node = helper.make_node(node_type, inputs=["x"], outputs=["y"])

        graph = helper.make_graph(
            [pool_node],
            "global_pooling_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="global_pooling_test")
        verify_with_ort(
            model,
            [x_shape],
            [out_shape],
            use_vm=False,
            target=target,
            dev=dev,
        )

    # Test each pooling mode across all N-D inputs.
    for mode in ["average", "max"]:
        # 1D Pooling (NCW)
        verify_global_pooling([1, 8, 8], mode)
        verify_global_pooling([4, 1, 4], mode)
        # 2D Pooling (NCHW)
        verify_global_pooling([1, 8, 8, 8], mode)
        verify_global_pooling([4, 1, 6, 4], mode)
        # 3D Pooling (NCDHW)
        verify_global_pooling([1, 8, 6, 8, 8], mode)
        verify_global_pooling([4, 1, 2, 6, 4], mode)




@tvm.testing.parametrize_targets
def test_mod(target, dev):
    """test_mod"""

    def verify_mod(x_shape, y_shape, fmod, out_shape, dtype="float32"):
        x_np = np.random.uniform(-100.0, 100.0, x_shape).astype(dtype)
        y_np = np.random.uniform(-100.0, 100.0, y_shape).astype(dtype)
        y_np = np.where(y_np == 0, 1, y_np)  # remove 0's to avoid division by zero error

        mod_node = helper.make_node("Mod", inputs=["x", "y"], outputs=["z"], fmod=fmod)

        onnx_dtype = TensorProto.FLOAT if dtype == "float32" else TensorProto.INT32
        graph = helper.make_graph(
            [mod_node],
            "mod_test",
            inputs=[
                helper.make_tensor_value_info("x", onnx_dtype, list(x_shape)),
                helper.make_tensor_value_info("y", onnx_dtype, list(y_shape)),
            ],
            outputs=[helper.make_tensor_value_info("z", onnx_dtype, list(out_shape))],
        )
        model = helper.make_model(graph, producer_name="mod_test")
        verify_with_ort_with_inputs(model, [x_np, y_np], [out_shape], target=target, dev=dev)

    # Mod
    verify_mod(
        x_shape=[1, 32, 32], y_shape=[1, 1, 32], fmod=0, out_shape=(1, 32, 32), dtype="int32"
    )
    verify_mod(
        x_shape=[1, 32, 32, 32],
        y_shape=[1, 32, 32, 32],
        fmod=0,
        out_shape=(1, 32, 32, 32),
        dtype="int32",
    )

    # fmod
    verify_mod(
        x_shape=[1, 32, 32], y_shape=[1, 32, 32], fmod=1, out_shape=(1, 32, 32), dtype="int32"
    )
    verify_mod(x_shape=[1, 1, 32, 32], y_shape=[1, 32, 32, 32], fmod=1, out_shape=(1, 32, 32, 32))
    verify_mod(x_shape=[1, 32, 32, 32], y_shape=[1, 1, 32, 32], fmod=1, out_shape=(1, 32, 32, 32))
    verify_mod(
        x_shape=[1, 32, 32, 32],
        y_shape=[1, 32, 32, 32],
        fmod=1,
        out_shape=(1, 32, 32, 32),
        dtype="int32",
    )
    verify_mod(x_shape=[1, 32, 32, 32], y_shape=[1, 32, 32, 32], fmod=1, out_shape=(1, 32, 32, 32))


@tvm.testing.parametrize_targets
def test_xor(target, dev):
    """test_xor"""

    def verify_xor(x_shape, y_shape):
        x_np = np.random.choice(a=[False, True], size=x_shape).astype("bool")
        y_np = np.random.choice(a=[False, True], size=y_shape).astype("bool")

        np_out = np.logical_xor(x_np, y_np)
        out_shape = np_out.shape

        xor_node = helper.make_node("Xor", inputs=["x", "y"], outputs=["z"])

        onnx_dtype = TensorProto.BOOL
        graph = helper.make_graph(
            [xor_node],
            "xor_test",
            inputs=[
                helper.make_tensor_value_info("x", onnx_dtype, list(x_shape)),
                helper.make_tensor_value_info("y", onnx_dtype, list(y_shape)),
            ],
            outputs=[helper.make_tensor_value_info("z", onnx_dtype, list(out_shape))],
        )
        model = helper.make_model(graph, producer_name="xor_test")
        verify_with_ort_with_inputs(model, [x_np, y_np], [out_shape], target=target, dev=dev)

    # XOR
    verify_xor(x_shape=[1, 32, 32], y_shape=[1, 32, 32])

    # Xor broadcast
    verify_xor(x_shape=[1, 32, 32], y_shape=[1, 1, 32])


@tvm.testing.parametrize_targets
def test_max_roi_pool(target, dev):
    """test_max_roi_pool"""

    def verify_max_roi_pool(x_shape, rois_shape, pooled_shape, spatial_scale, out_shape):
        if spatial_scale is None:
            pool_node = helper.make_node(
                "MaxRoiPool", inputs=["x", "rois"], outputs=["y"], pooled_shape=pooled_shape
            )
        else:
            pool_node = helper.make_node(
                "MaxRoiPool",
                inputs=["x", "rois"],
                outputs=["y"],
                pooled_shape=pooled_shape,
                spatial_scale=spatial_scale,
            )

        graph = helper.make_graph(
            [pool_node],
            "pool_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape)),
                helper.make_tensor_value_info("rois", TensorProto.FLOAT, list(rois_shape)),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="pool_test")
        verify_with_ort(model, [x_shape, rois_shape], [out_shape], target=target, dev=dev)

    verify_max_roi_pool(
        x_shape=[1, 3, 6, 6],
        rois_shape=[3, 5],
        pooled_shape=[1, 1],
        spatial_scale=None,
        out_shape=[3, 3, 1, 1],
    )

    verify_max_roi_pool(
        x_shape=[1, 3, 10, 10],
        rois_shape=[4, 5],
        pooled_shape=[2, 2],
        spatial_scale=2.0,
        out_shape=[4, 3, 2, 2],
    )


@tvm.testing.parametrize_targets
def test_lppool(target, dev):
    """test_lppool"""

    def verify_lppool(x_shape, kernel_shape, p, strides, pads, out_shape, auto_pad="NOTSET"):
        kwargs = {}
        if p is not None:
            kwargs["p"] = p
        if pads is None:
            pool_node = helper.make_node(
                "LpPool",
                inputs=["x"],
                outputs=["y"],
                kernel_shape=kernel_shape,
                auto_pad=auto_pad,
                strides=strides,
                **kwargs,
            )
        else:
            pool_node = helper.make_node(
                "LpPool",
                inputs=["x"],
                outputs=["y"],
                kernel_shape=kernel_shape,
                pads=pads,
                strides=strides,
                **kwargs,
            )

        graph = helper.make_graph(
            [pool_node],
            "lppool_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="lppool_test")
        verify_with_ort(
            model,
            [x_shape],
            [out_shape],
            use_vm=True,
            target=target,
            dev=dev,
        )

    # Pool1D
    verify_lppool(
        x_shape=[1, 1, 32], kernel_shape=[3], p=2, strides=[1], pads=[1, 1], out_shape=[1, 1, 32]
    )

    # Pool2D
    verify_lppool(
        x_shape=[1, 1, 32, 32],
        kernel_shape=[3, 3],
        p=2,
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        out_shape=[1, 1, 32, 32],
    )

    # Pool1D with stride
    verify_lppool(
        x_shape=[1, 1, 32], kernel_shape=[3], p=2, strides=[2], pads=[1, 1], out_shape=[1, 1, 16]
    )

    # Pool2D with stride
    verify_lppool(
        x_shape=[1, 1, 32, 32],
        kernel_shape=[3, 3],
        p=2,
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        out_shape=[1, 1, 16, 16],
    )

    # Pool1D with stride and autopadding
    verify_lppool(
        x_shape=[1, 1, 32],
        kernel_shape=[3],
        p=2,
        strides=[2],
        pads=None,
        out_shape=[1, 1, 16],
        auto_pad="SAME_UPPER",
    )

    # Pool2D with stride and autopadding
    verify_lppool(
        x_shape=[1, 1, 32, 32],
        kernel_shape=[3, 3],
        p=2,
        strides=[2, 2],
        pads=None,
        out_shape=[1, 1, 16, 16],
        auto_pad="SAME_UPPER",
    )

    # Pool2D with empty stride
    verify_lppool(
        x_shape=[1, 3, 32, 32],
        kernel_shape=[2, 2],
        p=4,
        strides=None,
        pads=None,
        out_shape=[1, 3, 32, 32],
        auto_pad="SAME_LOWER",
    )

    # Pool3D with stride
    verify_lppool(
        x_shape=[1, 1, 32, 32, 32],
        kernel_shape=[3, 3, 3],
        p=2,
        strides=[2, 2, 2],
        pads=[1, 1, 1, 1, 1, 1],
        out_shape=[1, 1, 16, 16, 16],
    )

    # Pool3D with stride and autopadding
    verify_lppool(
        x_shape=[1, 1, 32, 32, 32],
        kernel_shape=[3, 3, 3],
        p=2,
        strides=[2, 2, 2],
        pads=None,
        out_shape=[1, 1, 16, 16, 16],
        auto_pad="SAME_UPPER",
    )
    # Pool2D with empty p
    verify_lppool(
        x_shape=[1, 1, 32, 32],
        kernel_shape=[3, 3],
        p=None,
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        out_shape=[1, 1, 32, 32],
    )


def verify_global_lppool(x_shape, p, out_shape, target, dev):
    """verify_global_lppool"""
    pool_node = helper.make_node(
        "GlobalLpPool",
        inputs=["x"],
        outputs=["y"],
        p=p,
    )

    graph = helper.make_graph(
        [pool_node],
        "global_lppool_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))],
    )

    model = helper.make_model(graph, producer_name="global_lppool_test")
    verify_with_ort(model, [x_shape], out_shape, use_vm=True, target=target, dev=dev)


@tvm.testing.parametrize_targets
def test_global_lppool(target, dev):
    """test_global_lppool"""
    # LpPool1D
    verify_global_lppool(x_shape=[1, 15, 16], p=2, out_shape=[1, 15, 1], target=target, dev=dev)

    # LpPool2D
    verify_global_lppool(
        x_shape=[1, 15, 32, 32], p=2, out_shape=[1, 15, 1, 1], target=target, dev=dev
    )

    # LpPool2D
    verify_global_lppool(
        x_shape=[1, 15, 32, 32], p=3, out_shape=[1, 15, 1, 1], target=target, dev=dev
    )

    # LpPool3D
    verify_global_lppool(
        x_shape=[1, 15, 3, 32, 32], p=2, out_shape=[1, 15, 1, 1, 1], target=target, dev=dev
    )


def verify_rnn(
    seq_length,
    batch_size,
    input_size,
    hidden_size,
    rnn_type="LSTM",
    use_bias=False,
    activations=None,
    alphas=None,
    betas=None,
    use_initial_state=False,
    use_peep=False,
    linear_before_reset=False,
    directions=1,
    layout=0,
    rtol=1e-5,
    atol=1e-5,
    target=None,
    dev=None,
    use_sequence_lens=False,
):
    """verify_rnn"""
    if rnn_type == "RNN":
        multiplier = 1
    elif rnn_type == "LSTM":
        multiplier = 4
    elif rnn_type == "GRU":
        multiplier = 3
    else:
        raise NotImplementedError(f"{rnn_type} RNNs not yet supported.")

    if directions not in [1, 2]:
        raise ValueError(f"Direction should be either 1 or 2 (for bidirectional LSTMs)")

    def get_inputs():
        input_names = []
        input_values = []
        input_tensors = []

        def register(np_arr, name, shape=None):
            input_values.append(np_arr)
            input_names.append(name)

            # Map of numpy dtypes to the protobuf equivalent
            dtype_map = {
                "float32": TensorProto.FLOAT,
                "int32": TensorProto.INT32,
                "int8": TensorProto.INT8,
            }

            if np_arr.dtype.name not in dtype_map:
                raise ValueError(f"Unknown dtype we don't know how to handle {np.dtype.name}")
            if shape is None:
                shape = list(np_arr.shape)
            proto_type = dtype_map[np_arr.dtype.name]
            input_tensors.append(helper.make_tensor_value_info(name, proto_type, shape))

        if layout == 1:
            x_np = np.random.uniform(size=(batch_size, seq_length, input_size)).astype("float32")
        else:
            x_np = np.random.uniform(size=(seq_length, batch_size, input_size)).astype("float32")
        w_np = np.random.uniform(size=(directions, multiplier * hidden_size, input_size)).astype(
            "float32"
        )
        r_np = np.random.uniform(size=(directions, multiplier * hidden_size, hidden_size)).astype(
            "float32"
        )
        register(x_np, "X")
        register(w_np, "W")
        register(r_np, "R")

        if use_bias:
            b_np = np.random.uniform(size=(directions, multiplier * 2 * hidden_size)).astype(
                "float32"
            )
            register(b_np, "B")

        if use_sequence_lens:
            sequence_np = np.random.uniform(0, seq_length, size=(batch_size)).astype("int32")
            register(sequence_np, "sequence_lens")

        if use_initial_state:
            assert use_bias is True, "Initial states must have bias specified."

            if not use_sequence_lens:
                sequence_np = np.repeat(seq_length, batch_size).astype("int32")
                register(sequence_np, "sequence_lens")

            if layout == 1:
                initial_h_np = np.random.uniform(size=(batch_size, directions, hidden_size)).astype(
                    "float32"
                )
            else:
                initial_h_np = np.random.uniform(size=(directions, batch_size, hidden_size)).astype(
                    "float32"
                )
            register(initial_h_np, "initial_h")

            if rnn_type == "LSTM":
                if layout == 1:
                    initial_c_np = np.random.uniform(
                        size=(batch_size, directions, hidden_size)
                    ).astype("float32")
                else:
                    initial_c_np = np.random.uniform(
                        size=(directions, batch_size, hidden_size)
                    ).astype("float32")
                register(initial_c_np, "initial_c")

        if use_peep and rnn_type == "LSTM":
            assert use_initial_state is True, "Peepholes require initial state to be specified."
            p_np = np.random.uniform(size=(directions, 3 * hidden_size)).astype("float32")
            register(p_np, "P")

        return input_names, input_tensors, input_values

    input_names, input_tensors, input_values = get_inputs()

    def get_outputs():
        output_names = []
        graph_outputs = []
        output_shapes = []

        def register(name, shape, proto_type):
            output_names.append(name)
            graph_outputs.append(helper.make_tensor_value_info(name, proto_type, list(shape)))
            output_shapes.append(list(shape))

        if layout == 1:
            register("Y", [directions, seq_length, batch_size, hidden_size], TensorProto.FLOAT)
            register("Y_h", [batch_size, directions, hidden_size], TensorProto.FLOAT)
        else:
            register("Y", [seq_length, directions, batch_size, hidden_size], TensorProto.FLOAT)
            register("Y_h", [directions, batch_size, hidden_size], TensorProto.FLOAT)

        if rnn_type == "LSTM":
            if layout == 1:
                register("Y_c", [batch_size, directions, hidden_size], TensorProto.FLOAT)
            else:
                register("Y_c", [directions, batch_size, hidden_size], TensorProto.FLOAT)

        return output_names, graph_outputs, output_shapes

    output_names, graph_outputs, output_shapes = get_outputs()

    rnn_node = helper.make_node(
        rnn_type, inputs=input_names, outputs=output_names, hidden_size=hidden_size
    )
    if activations is not None:
        activations_attr = helper.make_attribute("activations", activations)
        rnn_node.attribute.append(activations_attr)
    if directions == 2:
        direction_attr = helper.make_attribute("direction", "bidirectional")
        rnn_node.attribute.append(direction_attr)
    if alphas is not None:
        alphas_attr = helper.make_attribute("activation_alpha", alphas)
        rnn_node.attribute.append(alphas_attr)
    if betas is not None:
        betas_attr = helper.make_attribute("activation_beta", betas)
        rnn_node.attribute.append(betas_attr)
    if linear_before_reset and rnn_type == "GRU":
        lbr_attr = helper.make_attribute("linear_before_reset", 1)
        rnn_node.attribute.append(lbr_attr)
    if layout == 1:
        layout_attr = helper.make_attribute("layout", 1)
        rnn_node.attribute.append(layout_attr)

    graph = helper.make_graph([rnn_node], "rnn_test", inputs=input_tensors, outputs=graph_outputs)

    model = helper.make_model(graph, producer_name="rnn_test")

    verify_with_ort_with_inputs(
        model, input_values, output_shapes, atol=atol, rtol=rtol, target=target, dev=dev
    )


def verify_rnn_helper(target, dev, rnn_type):
    num_activations = 1
    if rnn_type == "GRU":
        num_activations = 2
    elif rnn_type == "LSTM":
        num_activations = 3

    for directions in [1, 2]:
        # No bias.
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=False,
            rnn_type=rnn_type,
            directions=directions,
            target=target,
            dev=dev,
        )
        # large batch.
        verify_rnn(
            seq_length=4,
            batch_size=8,
            input_size=16,
            hidden_size=32,
            use_bias=True,
            rnn_type=rnn_type,
            directions=directions,
            target=target,
            dev=dev,
        )
        # Non power of two.
        verify_rnn(
            seq_length=3,
            batch_size=3,
            input_size=16,
            hidden_size=40,
            use_bias=True,
            rnn_type=rnn_type,
            directions=directions,
            target=target,
            dev=dev,
        )
        # Long sequence.
        verify_rnn(
            seq_length=8,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=True,
            rnn_type=rnn_type,
            directions=directions,
            target=target,
            dev=dev,
        )
        # Large hidden.
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=128,
            use_bias=True,
            rnn_type=rnn_type,
            directions=directions,
            target=target,
            dev=dev,
        )
        # Large input.
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=64,
            hidden_size=32,
            use_bias=True,
            rnn_type=rnn_type,
            directions=directions,
            target=target,
            dev=dev,
        )

        # Different activation testing.
        # Default value hardsigmoid.
        # TODO: onnxruntime <= v1.12.0 has wrong default value of all activation functions
        if rnn_type != "RNN":
            activations = ["HardSigmoid", "Tanh", "Tanh"][0:num_activations] * directions
            verify_rnn(
                seq_length=2,
                batch_size=1,
                input_size=16,
                hidden_size=32,
                use_bias=False,
                activations=activations,
                rnn_type=rnn_type,
                directions=directions,
                target=target,
                dev=dev,
            )
        # Multiple parametrized activations.
        activations = ["HardSigmoid", "LeakyRelu", "Tanh"][0:num_activations] * directions
        alphas = [2.0, 0.5, 0.0][0:num_activations] * directions
        betas = [0.3, 0.0, 0.0][0:num_activations] * directions
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=False,
            activations=activations,
            alphas=alphas,
            betas=betas,
            rnn_type=rnn_type,
            directions=directions,
            target=target,
            dev=dev,
        )
        # All parametrized with new Affine activation.
        activations = ["Affine", "LeakyRelu", "HardSigmoid"][0:num_activations] * directions
        alphas = [0.8, 2.0, 0.5][0:num_activations] * directions
        betas = [0.0, 0.3, 0.0][0:num_activations] * directions
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=False,
            activations=activations,
            alphas=alphas,
            betas=betas,
            rnn_type=rnn_type,
            directions=directions,
            target=target,
            dev=dev,
        )

        # Testing with initial state
        verify_rnn(
            seq_length=2,
            batch_size=1,
            input_size=16,
            hidden_size=32,
            use_bias=True,
            use_initial_state=True,
            rnn_type=rnn_type,
            directions=directions,
            target=target,
            dev=dev,
        )

        # Testing layout
        # TODO: onnxruntime <= 1.12.0 doesn't support layout == 1
        # verify_rnn(
        #     seq_length=2,
        #     batch_size=1,
        #     input_size=16,
        #     hidden_size=32,
        #     use_bias=True,
        #     rnn_type="RNN",
        #     directions=directions,
        #     layout=1,
        #     target=target,
        #     dev=dev,
        # )

        # Testing with initial state
        if rnn_type == "GRU":
            verify_rnn(
                seq_length=2,
                batch_size=1,
                input_size=16,
                hidden_size=32,
                use_bias=True,
                use_initial_state=True,
                rnn_type=rnn_type,
                directions=directions,
                target=target,
                dev=dev,
                use_sequence_lens=True,
            )
            verify_rnn(
                seq_length=8,
                batch_size=8,
                input_size=16,
                hidden_size=32,
                use_bias=True,
                use_initial_state=True,
                rnn_type=rnn_type,
                directions=directions,
                target=target,
                dev=dev,
                use_sequence_lens=True,
            )

        # Testing with peepholes
        if rnn_type == "LSTM":
            verify_rnn(
                seq_length=2,
                batch_size=1,
                input_size=16,
                hidden_size=32,
                use_bias=True,
                use_initial_state=True,
                use_peep=True,
                rnn_type="LSTM",
                directions=directions,
                target=target,
                dev=dev,
            )


@tvm.testing.parametrize_targets
def test_rnn(target, dev):
    verify_rnn_helper(target, dev, "RNN")


@tvm.testing.parametrize_targets
def test_lstm(target, dev):
    verify_rnn_helper(target, dev, "LSTM")


@tvm.testing.parametrize_targets
def test_gru(target, dev):
    verify_rnn_helper(target, dev, "GRU")


@tvm.testing.parametrize_targets
def test_resize(target, dev):
    """test_resize"""

    def verify(ishape, oshape, scales, mode, coord_trans="asymmetric", alpha=0.5, exclude=False):
        nodes = [
            make_constant_node("roi", onnx.TensorProto.FLOAT, (0,), []),
            make_constant_node("scales", onnx.TensorProto.FLOAT, (len(scales),), scales),
        ]
        input_names = ["X", "roi", "scales"]

        if oshape != []:
            nodes.append(
                make_constant_node("sizes", onnx.TensorProto.INT64, (len(oshape),), oshape)
            )
            input_names.append("sizes")
        nodes.append(
            helper.make_node(
                "Resize",
                inputs=input_names,
                outputs=["Y"],
                mode=mode,
                coordinate_transformation_mode=coord_trans,
                cubic_coeff_a=alpha,
                exclude_outside=exclude,
            )
        )

        if oshape == []:
            oshape = [round(dim * scale) for (dim, scale) in zip(ishape, scales)]
        graph = helper.make_graph(
            nodes,
            "resize_test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, ishape)],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, oshape)],
        )

        model = helper.make_model(graph, producer_name="resize_test")

        verify_with_ort(
            model,
            [ishape],
            [oshape],
            use_vm=True,
            opset=11,
            freeze_params=True,
            target=target,
            dev=dev,
        )

    for ndim in [1, 2, 3]:
        method = "nearest"
        for coord_trans in ["asymmetric", "align_corners", "half_pixel"]:
            # upsampling
            verify([1, 16] + [32] * ndim, [1, 16] + [64] * ndim, [], method, coord_trans)
            # downsampling
            verify([1, 16] + [32] * ndim, [1, 16] + [16] * ndim, [], method, coord_trans)
            # scales are specified instead of sizes
            verify([1, 16] + [32] * ndim, [], [1, 1] + [0.5] * ndim, method, coord_trans)
            verify([1, 16] + [32] * ndim, [], [1, 1] + [2] * ndim, method, coord_trans)

        method = "linear"
        # upsampling
        verify([1, 16] + [32] * ndim, [1, 16] + [64] * ndim, [], method)
        # downsampling
        verify([1, 16] + [32] * ndim, [1, 16] + [16] * ndim, [], method)
        # scales are specified instead of sizes
        verify([1, 16] + [32] * ndim, [], [1, 1] + [0.5] * ndim, method)
        verify([1, 16] + [32] * ndim, [], [1, 1] + [2] * ndim, method)

        if ndim == 2:
            # ONNX Runtime only supports cubic interpolation for 2D images
            method = "cubic"
            for alpha in [0.5, 0.75]:
                for exclude in [True, False]:
                    # upsampling
                    verify(
                        [1, 16] + [32] * ndim,
                        [1, 16] + [64] * ndim,
                        [],
                        method,
                        alpha=alpha,
                        exclude=exclude,
                    )
                    # downsampling
                    verify(
                        [1, 16] + [32] * ndim,
                        [1, 16] + [16] * ndim,
                        [],
                        method,
                        alpha=alpha,
                        exclude=exclude,
                    )
                    # scales are specified instead of sizes
                    verify(
                        [1, 16] + [32] * ndim,
                        [],
                        [1, 1] + [0.5] * ndim,
                        method,
                        alpha=alpha,
                        exclude=exclude,
                    )
                    verify(
                        [1, 16] + [32] * ndim,
                        [],
                        [1, 1] + [2] * ndim,
                        method,
                        alpha=alpha,
                        exclude=exclude,
                    )

    def verify_opset_10(ishape, scales, mode):
        nodes = [
            make_constant_node("scales", onnx.TensorProto.FLOAT, (len(scales),), scales),
        ]
        input_names = ["X", "scales"]
        nodes.append(
            helper.make_node(
                "Resize",
                inputs=input_names,
                outputs=["Y"],
                mode=mode,
            )
        )

        oshape = [round(dim * scale) for (dim, scale) in zip(ishape, scales)]
        graph = helper.make_graph(
            nodes,
            "resize_test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, ishape)],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, oshape)],
        )

        model = helper.make_model(graph, producer_name="resize_test")
        verify_with_ort(
            model,
            [ishape],
            [oshape],
            use_vm=True,
            freeze_params=True,
            opset=10,
            target=target,
            dev=dev,
        )

    verify_opset_10([1, 16, 32, 32], [1, 1, 2, 2], "nearest")
    verify_opset_10([1, 16, 32, 32], [1, 1, 0.5, 0.5], "linear")


@tvm.testing.parametrize_targets
def test_nonzero(target, dev):
    """test_nonzero"""

    def verify_nonzero(indata, outdata, dtype):
        node = helper.make_node(
            "NonZero",
            inputs=["X"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [node],
            "nonzero_test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.INT64, list(indata.shape))],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.INT64, list(outdata.shape))],
        )

        model = helper.make_model(graph, producer_name="nonzero_test")

        verify_with_ort_with_inputs(
            model, [indata], dtype="int64", use_vm=True, opset=9, target=target, dev=dev
        )

    input_data = np.array([[1, 0], [1, 1]], dtype=np.int64)
    result = np.array((np.nonzero(input_data)))  # expected output [[0, 1, 1], [0, 0, 1]]
    verify_nonzero(input_data, result, dtype=np.int64)

    input_data = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]], dtype=np.int64)
    result = np.array((np.nonzero(input_data)))  # expected output [[0, 1, 2, 2], [0, 1, 0, 1]]
    verify_nonzero(input_data, result, dtype=np.int64)


@tvm.testing.parametrize_targets
def test_topk(target, dev):
    """test_topk"""

    def verify_topk(input_dims, k, axis=-1):
        output_dims = list(input_dims)
        output_dims[axis] = k

        node = helper.make_node("TopK", inputs=["X", "K"], outputs=["Values", "Indices"], axis=axis)

        graph = helper.make_graph(
            [node],
            "topk_test",
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, list(input_dims)),
                helper.make_tensor_value_info(
                    "K",
                    TensorProto.INT64,
                    [
                        1,
                    ],
                ),
            ],
            outputs=[
                helper.make_tensor_value_info("Values", TensorProto.FLOAT, output_dims),
                helper.make_tensor_value_info("Indices", TensorProto.INT64, output_dims),
            ],
        )

        model = helper.make_model(graph, producer_name="topk_test")

        indata = np.random.uniform(-10, 10, input_dims).astype(np.float32)
        verify_with_ort_with_inputs(
            model, [indata, np.array([k])], use_vm=True, target=target, dev=dev
        )

    for n in [12, 32]:
        for shape in [[n], [n, n], [n, n, n]]:
            for k in [1, 5, 10]:
                verify_topk(shape, k)

        verify_topk([n, n, n], 5, 0)
        verify_topk([n, n, n], 5, 1)
        verify_topk([n, n, n], 5, 2)


@tvm.testing.parametrize_targets
def test_roi_align(target, dev):
    """test_roi_align"""

    def verify_roi_align(
        input_dims,
        num_roi,
        output_height,
        output_width,
        sampling_ratio=0,
        spatial_scale=1.0,
        mode="avg",
    ):
        output_dims = [num_roi, input_dims[1], output_height, output_width]

        node = helper.make_node(
            "RoiAlign",
            coordinate_transformation_mode="output_half_pixel",
            inputs=["X", "rois", "batch_indices"],
            outputs=["Y"],
            mode=mode,
            output_height=output_height,
            output_width=output_width,
            sampling_ratio=sampling_ratio,
            spatial_scale=spatial_scale,
        )

        graph = helper.make_graph(
            [node],
            "roialign_test",
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, list(input_dims)),
                helper.make_tensor_value_info("rois", TensorProto.FLOAT, [num_roi, 4]),
                helper.make_tensor_value_info(
                    "batch_indices",
                    TensorProto.INT64,
                    [
                        num_roi,
                    ],
                ),
            ],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_dims)],
        )

        model = helper.make_model(graph, producer_name="roialign_test")

        np_data = np.random.uniform(size=input_dims).astype("float32")
        np_rois = np.random.uniform(size=[num_roi, 4]).astype("float32") * input_dims[2]
        np_batch_indices = np.random.randint(low=0, high=input_dims[0], size=num_roi)

        verify_with_ort_with_inputs(
            model,
            [np_data, np_rois, np_batch_indices],
            out_shape=[output_dims],
            target=target,
            dev=dev,
        )

    verify_roi_align((1, 4, 16, 16), 32, 7, 7, sampling_ratio=0, spatial_scale=1.0)
    verify_roi_align((4, 4, 16, 32), 32, 7, 7, sampling_ratio=0, spatial_scale=1.0)
    verify_roi_align((1, 8, 16, 16), 32, 7, 7, sampling_ratio=0, spatial_scale=1.0)
    verify_roi_align((1, 4, 8, 8), 32, 7, 7, sampling_ratio=0, spatial_scale=1.0)
    verify_roi_align((1, 4, 16, 16), 16, 5, 7, sampling_ratio=0, spatial_scale=1.0)
    verify_roi_align((1, 4, 16, 12), 8, 7, 3, sampling_ratio=0, spatial_scale=1.0)
    verify_roi_align((1, 4, 16, 16), 32, 7, 7, sampling_ratio=0, spatial_scale=0.5)
    verify_roi_align((3, 4, 12, 16), 32, 7, 7, sampling_ratio=0, spatial_scale=1.5)
    verify_roi_align((5, 4, 16, 14), 32, 7, 7, sampling_ratio=1, spatial_scale=1.0)
    verify_roi_align((1, 4, 16, 16), 32, 7, 7, sampling_ratio=2, spatial_scale=1.0)

    # ONNX implementation of roi_align with max mode is incorrect, so we don't compare outputs here.


@tvm.testing.parametrize_targets
def test_non_max_suppression(target, dev):
    """test_non_max_suppression"""

    def verify_nms(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, output_dims
    ):
        input_names = ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold"]
        input_nodes = [
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes.shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores.shape),
            helper.make_tensor_value_info(
                "max_output_boxes_per_class", TensorProto.INT64, max_output_boxes_per_class.shape
            ),
            helper.make_tensor_value_info("iou_threshold", TensorProto.FLOAT, iou_threshold.shape),
        ]
        inputs = [boxes, scores, max_output_boxes_per_class, iou_threshold]
        if score_threshold is not None:
            input_names.append("score_threshold")
            input_nodes.append(
                helper.make_tensor_value_info(
                    "score_threshold", TensorProto.FLOAT, score_threshold.shape
                )
            )
            inputs.append(score_threshold)
        node = helper.make_node(
            "NonMaxSuppression",
            inputs=input_names,
            outputs=["Y"],
            center_point_box=0,
        )

        graph = helper.make_graph(
            [node],
            "nms_test",
            inputs=input_nodes,
            outputs=[helper.make_tensor_value_info("Y", TensorProto.INT64, output_dims)],
        )

        model = helper.make_model(graph, producer_name="nms_test")

        verify_with_ort_with_inputs(model, inputs, use_vm=True, target=target, dev=dev)

    boxes = np.array(
        [
            [
                [0.0, 0.0, 0.3, 0.3],
                [0.0, 0.0, 0.4, 0.4],
                [0.0, 0.0, 0.5, 0.5],
                [0.5, 0.5, 0.9, 0.9],
                [0.5, 0.5, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.3, 0.3],
                [0.0, 0.0, 0.4, 0.4],
                [0.5, 0.5, 0.95, 0.95],
                [0.5, 0.5, 0.96, 0.96],
                [0.5, 0.5, 1.0, 1.0],
            ],
        ]
    ).astype("float32")

    scores = np.array(
        [
            [[0.1, 0.2, 0.6, 0.3, 0.9], [0.1, 0.2, 0.6, 0.3, 0.9]],
            [[0.1, 0.2, 0.6, 0.3, 0.9], [0.1, 0.2, 0.6, 0.3, 0.9]],
        ]
    ).astype("float32")
    max_output_boxes_per_class = np.array(2).astype("int64")
    iou_threshold = np.array(0.8).astype("float32")
    output_dims = [8, 3]
    verify_nms(boxes, scores, max_output_boxes_per_class, iou_threshold, None, output_dims)

    boxes = np.array(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0],
            ]
        ]
    ).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.4]).astype(np.float32)
    output_dims = [2, 3]
    verify_nms(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, output_dims
    )


@tvm.testing.parametrize_targets
def test_loop(target, dev):
    """test_loop"""

    def verify_cond_loop():
        y_in = helper.make_tensor_value_info("y_in", TensorProto.FLOAT, [1])
        y_out = helper.make_tensor_value_info("y_out", TensorProto.FLOAT, [1])
        scan_out = helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [1])
        cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
        cond_out = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
        iter_count = helper.make_tensor_value_info("iter_count", TensorProto.INT64, [])

        y = np.array([-2]).astype(np.float32)

        five_const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["five"],
            value=helper.make_tensor(
                name="const_tensor_five", data_type=TensorProto.FLOAT, dims=(), vals=[5]
            ),
        )

        iter_cast_node = helper.make_node(
            "Cast", inputs=["iter_count"], outputs=["iter_cast"], to=onnx.TensorProto.FLOAT
        )

        y_add_node = helper.make_node("Add", inputs=["y_in", "iter_cast"], outputs=["y_out"])

        less_node = helper.make_node("Less", inputs=["y_out", "five"], outputs=["cond_less"])

        squeeze_node = helper.make_node("Squeeze", inputs=["cond_less"], outputs=["cond_squeeze"])

        cond_cast_node = helper.make_node(
            "Cast", inputs=["cond_squeeze"], outputs=["cond_out"], to=onnx.TensorProto.BOOL
        )

        scan_identity_node = helper.make_node("Identity", inputs=["y_out"], outputs=["scan_out"])

        loop_body = helper.make_graph(
            [
                five_const_node,
                iter_cast_node,
                y_add_node,
                less_node,
                squeeze_node,
                cond_cast_node,
                scan_identity_node,
            ],
            "loop_body",
            [iter_count, cond_in, y_in],
            [cond_out, y_out, scan_out],
        )

        loop_node = helper.make_node(
            "Loop",
            inputs=["trip_count", "cond", "y"],
            outputs=["res_y", "res_scan"],
            body=loop_body,
        )

        trip_count = np.array(5).astype(np.int64)
        _ = np.array([13]).astype(np.float32)
        cond = np.array(1).astype(bool)
        loop_graph = onnx.helper.make_graph(
            [loop_node],
            "loop_outer",
            inputs=[
                onnx.helper.make_tensor_value_info("trip_count", onnx.TensorProto.INT64, []),
                onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
                onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1]),
            ],
            outputs=[
                onnx.helper.make_tensor_value_info("res_y", onnx.TensorProto.FLOAT, [1]),
                onnx.helper.make_tensor_value_info("res_scan", onnx.TensorProto.FLOAT, [5, 1]),
            ],
        )
        loop_model = onnx.helper.make_model(loop_graph)

        # Set a high trip count so that condition trips first.
        trip_count = np.array(40).astype(np.int64)
        cond = np.array(1).astype(bool)
        input_vals = [trip_count, cond, y]
        verify_with_ort_with_inputs(
            loop_model,
            input_vals,
            use_vm=True,
            freeze_params=True,
            opset=11,
            target=target,
            dev=dev,
        )

    def verify_count_loop():
        y_in = helper.make_tensor_value_info("y_in", TensorProto.FLOAT, [])
        y_out = helper.make_tensor_value_info("y_out", TensorProto.FLOAT, [])
        scan_out = helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [])
        cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
        cond_out = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
        iter_count = helper.make_tensor_value_info("iter_count", TensorProto.INT64, [])

        y = np.array(-2).astype(np.float32)

        iter_cast_node = helper.make_node(
            "Cast", inputs=["iter_count"], outputs=["iter_cast"], to=onnx.TensorProto.FLOAT
        )

        y_add_node = helper.make_node("Add", inputs=["y_in", "iter_cast"], outputs=["y_out"])

        identity_node = helper.make_node("Identity", inputs=["cond_in"], outputs=["cond_out"])

        scan_identity_node = helper.make_node("Identity", inputs=["y_out"], outputs=["scan_out"])

        loop_body = helper.make_graph(
            [identity_node, iter_cast_node, y_add_node, scan_identity_node],
            "loop_body",
            [iter_count, cond_in, y_in],
            [cond_out, y_out, scan_out],
        )

        loop_node = helper.make_node(
            "Loop",
            inputs=["trip_count", "cond", "y"],
            outputs=["res_y", "res_scan"],
            body=loop_body,
        )

        trip_count = np.array(5).astype(np.int64)
        _ = np.array([13]).astype(np.float32)
        cond = np.array(1).astype(bool)
        loop_graph = onnx.helper.make_graph(
            [loop_node],
            "loop_outer",
            inputs=[
                onnx.helper.make_tensor_value_info("trip_count", onnx.TensorProto.INT64, []),
                onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
                onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, []),
            ],
            outputs=[
                onnx.helper.make_tensor_value_info("res_y", onnx.TensorProto.FLOAT, []),
                onnx.helper.make_tensor_value_info("res_scan", onnx.TensorProto.FLOAT, [5]),
            ],
        )
        loop_model = onnx.helper.make_model(loop_graph)

        trip_count = np.array(5).astype(np.int64)
        cond = np.array(1).astype(bool)
        input_vals = [trip_count, cond, y]
        verify_with_ort_with_inputs(
            loop_model,
            input_vals,
            use_vm=True,
            freeze_params=True,
            opset=11,
            target=target,
            dev=dev,
        )

    def verify_tensor_loop(shapeless_output=False):
        y_in = helper.make_tensor_value_info("y_in", TensorProto.FLOAT, [3, 3, 3, 3])
        y_out = helper.make_tensor_value_info("y_out", TensorProto.FLOAT, [3, 3, 3, 3])
        scan_out = helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [3, 3, 3, 3])
        cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
        cond_out = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
        iter_count = helper.make_tensor_value_info("iter_count", TensorProto.INT64, [])

        y = np.random.normal(size=[3, 3, 3, 3]).astype(np.float32)

        iter_cast_node = helper.make_node(
            "Cast", inputs=["iter_count"], outputs=["iter_cast"], to=onnx.TensorProto.FLOAT
        )

        y_add_node = helper.make_node("Add", inputs=["y_in", "iter_cast"], outputs=["y_out"])

        identity_node = helper.make_node("Identity", inputs=["cond_in"], outputs=["cond_out"])

        scan_identity_node = helper.make_node("Identity", inputs=["y_out"], outputs=["scan_out"])

        loop_body = helper.make_graph(
            [identity_node, iter_cast_node, y_add_node, scan_identity_node],
            "loop_body",
            [iter_count, cond_in, y_in],
            [cond_out, y_out, scan_out],
        )

        loop_node = helper.make_node(
            "Loop",
            inputs=["trip_count", "cond", "y"],
            outputs=["res_y", "res_scan"],
            body=loop_body,
        )

        trip_count = np.array(5).astype(np.int64)
        cond = np.array(1).astype(bool)

        # Allow testing of malformed nodes since pytorch likes to create these.
        if shapeless_output:
            scan_shape = None
        else:
            scan_shape = [5, 3, 3, 3, 3]

        loop_graph = onnx.helper.make_graph(
            [loop_node],
            "loop_outer",
            inputs=[
                onnx.helper.make_tensor_value_info("trip_count", onnx.TensorProto.INT64, []),
                onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
                onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [3, 3, 3, 3]),
            ],
            outputs=[
                onnx.helper.make_tensor_value_info("res_y", onnx.TensorProto.FLOAT, [3, 3, 3, 3]),
                onnx.helper.make_tensor_value_info("res_scan", onnx.TensorProto.FLOAT, scan_shape),
            ],
        )
        loop_model = onnx.helper.make_model(loop_graph)

        trip_count = np.array(5).astype(np.int64)
        cond = np.array(1).astype(bool)
        input_vals = [trip_count, cond, y]
        verify_with_ort_with_inputs(
            loop_model,
            input_vals,
            use_vm=True,
            freeze_params=True,
            opset=11,
            target=target,
            dev=dev,
        )

    # Test a loop that exits once a condition is met.
    verify_cond_loop()
    # Test a loop that exits after a fixed number of iterations with scalar outputs.
    verify_count_loop()
    # Test a loop that uses an array output.
    verify_tensor_loop()
    # Test a loop that is malformed and has no output shape defined.
    verify_tensor_loop(shapeless_output=True)


@tvm.testing.parametrize_targets
def test_if(target, dev):
    """test_if"""

    def verify_if(cond_array, num_outputs):
        # Given a bool scalar input cond.
        # return constant tensor x if cond is True, otherwise return constant tensor y.

        def append_constant_nodes(nodes, outputs, expected, name):
            outputs.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [5]))

            expected.append(np.random.randn(5).astype("float32"))

            nodes.append(
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[name],
                    value=numpy_helper.from_array(expected[-1]),
                )
            )

        if_outputs = []
        graph_outputs = []

        then_nodes, then_outs, then_expected = [], [], []
        else_nodes, else_outs, else_expected = [], [], []

        for i in range(num_outputs):
            append_constant_nodes(then_nodes, then_outs, then_expected, f"then_out{i}")
            append_constant_nodes(else_nodes, else_outs, else_expected, f"else_out{i}")

            if_outputs.append(f"res{i}")
            graph_outputs.append(
                onnx.helper.make_tensor_value_info(f"res{i}", onnx.TensorProto.FLOAT, [5]),
            )

        then_body = onnx.helper.make_graph(then_nodes, "then_body", [], then_outs)
        else_body = onnx.helper.make_graph(else_nodes, "else_body", [], else_outs)

        if_node = onnx.helper.make_node(
            "If", inputs=["cond"], outputs=if_outputs, then_branch=then_body, else_branch=else_body
        )

        if_graph = onnx.helper.make_graph(
            [if_node],
            "if_outer",
            inputs=[
                onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
            ],
            outputs=graph_outputs,
        )

        if_model = onnx.helper.make_model(if_graph)
        if cond_array:
            cond = np.array([1]).astype("bool")
        else:
            cond = np.array(1).astype("bool")
        correct_out = then_expected if cond else else_expected

        # TODO(jwfromm): Onnxruntime 1.0.0 is buggy with If statements. Replace this with
        # verify_with_ort once we update versions.
        tvm_out = get_tvm_output_with_vm(if_model, [cond], target, dev, freeze_params=True)
        if not isinstance(tvm_out, list):
            tvm_out = [tvm_out]
        for i, _ in enumerate(tvm_out):
            tvm.testing.assert_allclose(
                correct_out[i],
                tvm_out[i],  # pylint: disable=unnecessary-list-index-lookup
                rtol=1e-05,
                atol=1e-05,
            )

    # Confirm that if works with cond as an array or scalar.
    verify_if(cond_array=False, num_outputs=1)
    verify_if(cond_array=False, num_outputs=2)
    verify_if(cond_array=True, num_outputs=1)
    verify_if(cond_array=True, num_outputs=2)


@tvm.testing.parametrize_targets
def test_size(target, dev):
    """test_size"""

    def verify_size(indata):
        node = helper.make_node(
            "Size",
            inputs=["X"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [node],
            "size_test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.INT64, list(indata.shape))],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.INT64, [])],
        )

        model = helper.make_model(graph, producer_name="size_test")

        verify_with_ort_with_inputs(
            model, [indata], dtype="int64", use_vm=True, opset=11, target=target, dev=dev
        )

    input_data = np.array([[1, 0], [1, 1]], dtype=np.int64)
    verify_size(input_data)

    input_data = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]], dtype=np.int64)
    verify_size(input_data)


@tvm.testing.parametrize_targets
def test_maxunpool(target, dev):
    """test_maxunpool"""

    def verify_maxunpool(data, indices, kernel_shape, strides, output_shape=None, pads=None):
        input_names = ["xT", "xI"]
        input_info = [
            helper.make_tensor_value_info("xT", TensorProto.FLOAT, list(data.shape)),
            helper.make_tensor_value_info("xI", TensorProto.INT64, list(indices.shape)),
        ]
        input_values = [data, indices]
        if output_shape is not None:
            input_names.append("output_shape")
            input_info.append(
                helper.make_tensor_value_info(
                    "output_shape", TensorProto.INT64, list(output_shape.shape)
                )
            )
            input_values.append(output_shape)
        else:
            # Compute expected output shape
            output_shape = np.asarray(([1, 1] + list(strides))) * np.asarray(list(data.shape))
            output_shape += np.asarray(([0, 0] + list(kernel_shape))) - np.asarray(
                ([0, 0] + list(strides))
            )
            if pads is not None:
                output_shape -= np.asarray(
                    [0, 0] + list(np.sum(np.reshape(list(pads), [-1, 2]), axis=-1))
                )
        output_shape = [int(i) for i in output_shape]

        node = helper.make_node(
            "MaxUnpool", inputs=input_names, outputs=["y"], kernel_shape=kernel_shape
        )

        if pads is not None:
            pad_attr = helper.make_attribute("pads", pads)
            node.attribute.append(pad_attr)

        if strides is not None:
            strides_attr = helper.make_attribute("strides", strides)
            node.attribute.append(strides_attr)

        graph = helper.make_graph(
            [node],
            "maxunpool_test",
            inputs=input_info,
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )

        model = helper.make_model(graph, producer_name="size_test")

        verify_with_ort_with_inputs(
            model, input_values, use_vm=True, opset=11, target=target, dev=dev
        )

    # Basic test
    x_t = np.array([[[[5, 6], [7, 8]]]], dtype=np.float32)
    x_i = np.array([[[[0, 7], [13, 15]]]], dtype=np.int64)
    verify_maxunpool(x_t, x_i, [2, 2], strides=[2, 2])
    # Small stride
    verify_maxunpool(x_t, x_i, [2, 2], strides=[1, 1])
    # Big kernel
    verify_maxunpool(x_t, x_i, [3, 3], strides=[2, 2])
    # With output shape
    output_shape = np.array((1, 1, 5, 5), dtype=np.int64)
    verify_maxunpool(x_t, x_i, [2, 2], strides=[2, 2], output_shape=output_shape)
    # With explicit reverse padding
    pads = np.asarray([1, 1, 1, 1]).astype(np.int64)
    verify_maxunpool(x_t, x_i, [2, 2], strides=[2, 2], pads=pads)


@tvm.testing.parametrize_targets
def test_softplus(target, dev):
    """test_softplus"""

    def verify_softplus(indata):
        node = helper.make_node(
            "Softplus",
            inputs=["X"],
            outputs=["Y"],
        )

        graph = helper.make_graph(
            [node],
            "softplus_test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, list(indata.shape))],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(indata.shape))],
        )

        model = helper.make_model(graph, producer_name="softplus_test")

        verify_with_ort_with_inputs(
            model, [indata], dtype="float32", use_vm=True, opset=11, target=target, dev=dev
        )

    # Simple case with all signs.
    input_data = np.array([[-1, 0, 1]], dtype=np.float32)
    verify_softplus(input_data)
    # More fancy case.
    input_data = np.random.randn(1, 32, 32, 3).astype("float32")
    verify_softplus(input_data)


@tvm.testing.parametrize_targets
def test_cumsum(target, dev):
    """test_cumsum"""

    def verify_cumsum(indata, axis, exclusive=0, reverse=0, dtype="float32"):
        cumsum_node = onnx.helper.make_node(
            "CumSum",
            inputs=["X", "axis"],
            outputs=["Y"],
        )
        if exclusive != 0:
            exclusive_attr = helper.make_attribute("exclusive", exclusive)
            cumsum_node.attribute.append(exclusive_attr)
        if reverse != 0:
            reverse_attr = helper.make_attribute("reverse", reverse)
            cumsum_node.attribute.append(reverse_attr)
        nodes = [
            make_constant_node("axis", onnx.TensorProto.INT32, [1], [axis]),
            cumsum_node,
        ]
        if dtype == "float32":
            tensor_type = TensorProto.FLOAT
        else:
            tensor_type = TensorProto.INT32
            dtype = "int32"

        graph = helper.make_graph(
            nodes,
            "cumsum_test",
            inputs=[
                helper.make_tensor_value_info("X", tensor_type, list(indata.shape)),
            ],
            outputs=[helper.make_tensor_value_info("Y", tensor_type, list(indata.shape))],
        )

        model = helper.make_model(graph, producer_name="cumsum_test")

        verify_with_ort_with_inputs(
            model, [indata], dtype=dtype, use_vm=True, opset=11, target=target, dev=dev
        )

    data = (
        np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
            ]
        )
        .astype(np.float32)
        .reshape((3, 4))
    )

    verify_cumsum(data, 0)
    verify_cumsum(data, 1)
    verify_cumsum(data, 0, 1, 0)
    verify_cumsum(data, 1, 1, 0)
    verify_cumsum(data, 0, 0, 1)
    verify_cumsum(data, 1, 0, 1)
    verify_cumsum(data, 1, 1, 1)
    data = np.random.randn(1, 32, 32, 3).astype("float32")
    verify_cumsum(data, 1)
    data = np.random.randn(1, 32, 32, 3).astype("int32")
    verify_cumsum(data, 0, dtype="int32")
    verify_cumsum(data, 1, dtype="int32")
    verify_cumsum(data, 0, 1, 0, dtype="int32")
    verify_cumsum(data, 1, 1, 0, dtype="int32")
    verify_cumsum(data, 0, 0, 1, dtype="int32")
    verify_cumsum(data, 1, 0, 1, dtype="int32")
    verify_cumsum(data, 1, 1, 1, dtype="int32")


@tvm.testing.parametrize_targets
def test_eyelike(target, dev):
    """test_eyelike"""

    def verify_eyelike(indata, dynamic=False):
        node_list = []
        eyelike_inputs = ["X"]
        input_node_list = [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, list(indata.shape))
        ]
        input_list = [indata]

        if dynamic:
            input_node_list.append(
                helper.make_tensor_value_info("shape", TensorProto.INT64, [len(indata.shape)])
            )
            input_list.append(np.asarray(indata.shape))
            reshape_node = helper.make_node("Reshape", ["X", "shape"], ["X_dyn"])
            eyelike_inputs[0] = "X_dyn"
            node_list += [reshape_node]

        node = helper.make_node(
            "EyeLike",
            inputs=eyelike_inputs,
            outputs=["Y"],
        )
        node_list.append(node)

        graph = helper.make_graph(
            node_list,
            "eyelike_test",
            inputs=input_node_list,
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(indata.shape))],
        )

        model = helper.make_model(graph, producer_name="eyelike_test")
        verify_with_ort_with_inputs(
            model, input_list, dtype="float32", opset=9, target=target, dev=dev, use_vm=True
        )

    input_data = np.zeros((5, 5), dtype=np.float32)
    verify_eyelike(input_data)
    verify_eyelike(input_data, True)


# The following parametrized tests loads the tests that ONNX ships as
# serialized ONNX files, inputs, and outputs. The goal of this test
# is to ensure the ONNX importer is in line with the ONNX specification.
# To allow these tests to run in CI before all pass, a number of tests
# that are not yet supported are skipped.

onnx_test_node_dir = os.path.join(os.path.dirname(onnx.__file__), "backend", "test", "data", "node")

onnx_test_folders = sorted(
    dirname
    for dirname in os.listdir(onnx_test_node_dir)
    if dirname.startswith("test") and os.path.isdir(os.path.join(onnx_test_node_dir, dirname))
)

unsupported_onnx_tests = [
    "test_batchnorm_epsilon_training_mode",
    "test_batchnorm_example_training_mode",
    "test_bernoulli",
    "test_bernoulli_expanded",
    "test_bernoulli_double",
    "test_bernoulli_double_expanded",
    "test_bernoulli_seed",
    "test_bernoulli_seed_expanded",
    "test_blackmanwindow",
    "test_blackmanwindow_expanded",
    "test_blackmanwindow_symmetric",
    "test_blackmanwindow_symmetric_expanded",
    # the follow cast and castlike cases have lowering issues
    "test_cast_FLOAT_to_STRING",
    "test_cast_STRING_to_FLOAT",
    "test_castlike_FLOAT_to_STRING",
    "test_castlike_FLOAT_to_STRING_expanded",
    "test_castlike_STRING_to_FLOAT",
    "test_castlike_STRING_to_FLOAT_expanded",
    # the following cast and castlike cases segfault
    "test_cast_DOUBLE_to_FLOAT16",
    "test_castlike_DOUBLE_to_FLOAT16",
    "test_castlike_DOUBLE_to_FLOAT16_expanded",
    "test_convtranspose_autopad_same",
    "test_convtranspose_dilations",
    "test_cumsum_1d",
    "test_cumsum_1d_exclusive",
    "test_cumsum_1d_reverse",
    "test_cumsum_1d_reverse_exclusive",
    "test_cumsum_2d_axis_0",
    "test_cumsum_2d_axis_1",
    "test_cumsum_2d_negative_axis",
    "test_det_2d",
    "test_det_nd",
    "test_dropout_default",
    "test_dropout_default_mask",
    "test_dropout_default_mask_ratio",
    "test_dropout_default_ratio",
    "test_gru_batchwise",
    "test_hammingwindow",
    "test_hammingwindow_expanded",
    "test_hammingwindow_symmetric",
    "test_hammingwindow_symmetric_expanded",
    "test_hannwindow",
    "test_hannwindow_expanded",
    "test_hannwindow_symmetric",
    "test_hannwindow_symmetric_expanded",
    "test_identity_opt",
    "test_identity_sequence",
    "test_if_opt",
    "test_if_seq",
    "test_loop13_seq",
    "test_loop16_seq_none",
    "test_lstm_batchwise",
    "test_maxpool_with_argmax_2d_precomputed_pads",
    "test_maxpool_with_argmax_2d_precomputed_strides",
    "test_maxunpool_export_with_output_shape",
    "test_melweightmatrix",
    # This test fails llvm with a lowering error:
    "test_nllloss_NCd1d2d3_none_no_weight_negative_ii_expanded",
    "test_qlinearmatmul_3D",
    "test_range_float_type_positive_delta_expanded",
    "test_range_int32_type_negative_delta_expanded",
    "test_reduce_sum_do_not_keepdims_example",
    "test_reduce_sum_do_not_keepdims_random",
    "test_reduce_sum_keepdims_example",
    "test_reduce_sum_keepdims_random",
    "test_reduce_sum_negative_axes_keepdims_example",
    "test_reduce_sum_negative_axes_keepdims_random",
    "test_roialign_aligned_true",
    "test_sequence_insert_at_back",
    "test_sequence_insert_at_front",
    "test_sequence_map_add_1_sequence_1_tensor",
    "test_sequence_map_add_1_sequence_1_tensor_expanded",
    "test_sequence_map_add_2_sequences",
    "test_sequence_map_add_2_sequences_expanded",
    "test_sequence_map_extract_shapes",
    "test_sequence_map_extract_shapes_expanded",
    "test_sequence_map_identity_1_sequence",
    "test_sequence_map_identity_1_sequence_1_tensor",
    "test_sequence_map_identity_1_sequence_1_tensor_expanded",
    "test_sequence_map_identity_1_sequence_expanded",
    "test_sequence_map_identity_2_sequences",
    "test_sequence_map_identity_2_sequences_expanded",
    "test_simple_rnn_batchwise",
    "test_simple_rnn_defaults",
    "test_simple_rnn_with_initial_bias",
    "test_split_variable_parts_1d",
    "test_split_variable_parts_2d",
    "test_split_variable_parts_default_axis",
    "test_split_zero_size_splits",
    "test_stft",
    "test_stft_with_window",
    "test_strnormalizer_export_monday_casesensintive_lower",
    "test_strnormalizer_export_monday_casesensintive_nochangecase",
    "test_strnormalizer_export_monday_casesensintive_upper",
    "test_strnormalizer_export_monday_empty_output",
    "test_strnormalizer_export_monday_insensintive_upper_twodim",
    "test_strnormalizer_nostopwords_nochangecase",
    "test_tfidfvectorizer_tf_batch_onlybigrams_skip0",
    "test_tfidfvectorizer_tf_batch_onlybigrams_skip5",
    "test_tfidfvectorizer_tf_batch_uniandbigrams_skip5",
    "test_tfidfvectorizer_tf_only_bigrams_skip0",
    "test_tfidfvectorizer_tf_onlybigrams_levelempty",
    "test_tfidfvectorizer_tf_onlybigrams_skip5",
    "test_tfidfvectorizer_tf_uniandbigrams_skip5",
    "test_training_dropout",
    "test_training_dropout_default",
    "test_training_dropout_default_mask",
    "test_training_dropout_mask",
    "test_training_dropout_zero_ratio",
    "test_training_dropout_zero_ratio_mask",
    "test_tril_zero",
    "test_triu_zero",
    "test_unique_sorted_with_axis",
    "test_unique_sorted_with_axis_3d",
    "test_unique_sorted_with_negative_axis",
    "test_upsample_nearest",
]


target_skips = {
    "cuda": [
        "test_range_float_type_positive_delta_expanded",
        "test_range_int32_type_positive_delta_expanded",
        "test_mod_mixed_sign_float16",
        "test_qlinearconv",
        "test_qlinearmatmul",
        "test_resize_upsample_sizes_nearest",
    ]
}


def _load_proto(proto_filename, target_list, model_type_proto):
    with open(proto_filename, "rb") as fin:
        protobuf_content = fin.read()
        if model_type_proto.HasField("sequence_type"):
            sequence = onnx.SequenceProto()
            sequence.ParseFromString(protobuf_content)
            target_list.append(numpy_helper.to_list(sequence))
        elif model_type_proto.HasField("tensor_type"):
            tensor = onnx.TensorProto()
            tensor.ParseFromString(protobuf_content)
            target_list.append(numpy_helper.to_array(tensor))
        elif model_type_proto.HasField("optional_type"):
            optional = onnx.OptionalProto()
            optional.ParseFromString(protobuf_content)
            target_list.append(numpy_helper.to_optional(optional))
        else:
            raise ValueError(
                "Loading proto of that specific type (Map/Sparse Tensor) is currently not supported"
            )


@pytest.mark.parametrize("onnx_test", onnx_test_folders)
@tvm.testing.parametrize_targets
def test_onnx_nodes(target, dev, onnx_test):
    """test_onnx_nodes"""
    if platform.machine() == "aarch64" and onnx_test == "test_resize_upsample_sizes_nearest":
        pytest.skip("Currently failing on AArch64")

    target_kind = tvm.target.Target(target).kind.name

    if onnx_test in unsupported_onnx_tests:
        pytest.skip(f"Onnx test '{onnx_test}' not yet supported by TVM")

    target_specific_skips = target_skips.get(target_kind, [])
    if onnx_test in target_specific_skips:
        pytest.skip(f"Onnx test '{onnx_test}' not yet supported by TVM on {target_kind} targets")

    test_dir = os.path.join(onnx_test_node_dir, onnx_test)

    atol = 1e-5
    rtol = 1e-5
    if "roialign" in test_dir:
        # for some reason the ONNX test crops the
        # roialign results to 4 decimal places
        atol = 1e-4

    if "to_BFLOAT16" in test_dir:
        # the tolerance here is for the comparison in uint16 space, but is not as significant
        # of a delta in bfloat16 space because it's representing the mantissa being off by 1
        atol = 1

    if "_sce_" in test_dir:
        # complicated loss functions like SoftmaxCrossEntropy can have minor variations
        # in accuracy depending on implementation
        atol = 1e-4

    if "bicubic" in test_dir:
        # satisfies onnx precision for bicubic interpolation
        atol = 1e-4

    if "dft" in test_dir:
        atol = 1e-3

    model = onnx.load(os.path.join(test_dir, "model.onnx"))
    for test_data_dir in glob.glob(os.path.join(test_dir, "test_data_set*")):
        inputs = []
        n_inputs = len(glob.glob(os.path.join(test_data_dir, "input_*.pb")))
        for i in range(n_inputs):
            input_file = os.path.join(test_data_dir, f"input_{i}.pb")
            _load_proto(input_file, inputs, model.graph.input[i].type)

        outputs = []
        n_outputs = len(glob.glob(os.path.join(test_data_dir, "output_*.pb")))
        for i in range(n_outputs):
            output_file = os.path.join(test_data_dir, f"output_{i}.pb")
            _load_proto(output_file, outputs, model.graph.output[i].type)

    tvm_val = get_tvm_output_with_vm(model, inputs, target, dev)
    if len(outputs) == 1:
        tvm.testing.assert_allclose(outputs[0], tvm_val, rtol=rtol, atol=atol)
    else:
        for output, val in zip(outputs, tvm_val):
            tvm.testing.assert_allclose(output, val, rtol=rtol, atol=atol)


def test_wrong_input():
    """test_wrong_input"""
    node = helper.make_node(
        "Softplus",
        inputs=["X"],
        outputs=["Y"],
    )

    graph = helper.make_graph(
        [node],
        "softplus_test",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, list([5]))],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, list([5]))],
    )
    model = helper.make_model(graph, producer_name="softplus_test")

    # Check that the graph can import correctly with proper shape definitions.
    correct_shape_dict = {"X": [5]}
    relay.frontend.from_onnx(model, shape=correct_shape_dict)

    # Check that an assertion is triggered when an input not in the graph is provided.
    wrong_shape_dict = {"Z": [5]}
    with pytest.raises(AssertionError):
        relay.frontend.from_onnx(model, shape=wrong_shape_dict)





@tvm.testing.parametrize_targets
def test_reverse_sequence(target, dev):
    """test_reverse_sequence"""

    def verify_reverse_sequence(x, sequence_lens, batch_axis, time_axis):
        node = onnx.helper.make_node(
            "ReverseSequence",
            inputs=["x", "sequence_lens"],
            outputs=["y"],
            time_axis=time_axis,
            batch_axis=batch_axis,
        )

        graph = helper.make_graph(
            [node],
            "reverse_sequence_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
                helper.make_tensor_value_info(
                    "sequence_lens", TensorProto.INT64, list(sequence_lens.shape)
                ),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(x.shape))],
        )

        model = helper.make_model(graph, producer_name="reverse_sequence_test")
        verify_with_ort_with_inputs(model, [x, sequence_lens], [x.shape], target=target, dev=dev)

    x = np.array(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        dtype=np.float32,
    )
    sequence_lens = np.array([1, 2, 3, 4], dtype=np.int64)
    verify_reverse_sequence(x, sequence_lens, 0, 1)

    sequence_lens = np.array([4, 3, 2, 1], dtype=np.int64)
    verify_reverse_sequence(x, sequence_lens, 1, 0)


@pytest.mark.parametrize("op_name", ["Gelu", "FastGelu"], scope="session")
@pytest.mark.parametrize("data_type", ["float16", "float32"], scope="session")
@tvm.testing.parametrize_targets
def test_gelu(target, dev, data_type, op_name):
    """test_gelu"""
    dtype = np.dtype(data_type)
    tensor_type = mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    absolute_tolerance = 1e-3 if data_type == "float16" else 1e-5

    def verify_gelu(x):
        node = onnx.helper.make_node(
            op_name,
            inputs=["x"],
            outputs=["y"],
            domain="com.microsoft",
        )

        graph = helper.make_graph(
            [node],
            f"{op_name}_test",
            inputs=[helper.make_tensor_value_info("x", tensor_type, list(x.shape))],
            outputs=[helper.make_tensor_value_info("y", tensor_type, list(x.shape))],
        )

        model = helper.make_model(graph, producer_name=f"{op_name}_test")
        verify_with_ort_with_inputs(
            model, [x], [x.shape], atol=absolute_tolerance, dtype=data_type, target=target, dev=dev
        )

    x = np.array([-1.0, 0, 1.0, 100.0, -100.0, 1000.0, -1000.0], dtype=dtype)
    verify_gelu(x)
    x = np.array([[1, 2], [3, 4]], dtype=dtype)
    verify_gelu(x)


@pytest.mark.parametrize("op_name", ["BiasGelu", "FastGelu"], scope="session")
@pytest.mark.parametrize("data_type", ["float16", "float32"], scope="session")
@tvm.testing.parametrize_targets
def test_biasgelu(target, dev, data_type, op_name):
    """test_biasgelu"""
    dtype = np.dtype(data_type)
    tensor_type = mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    absolute_tolerance = 1e-2 if data_type == "float16" else 1e-5

    def verify_biasgelu(x, bias):
        node = onnx.helper.make_node(
            op_name,
            inputs=["x", "bias"],
            outputs=["y"],
            domain="com.microsoft",
        )

        graph = helper.make_graph(
            [node],
            f"{op_name}_test",
            inputs=[
                helper.make_tensor_value_info("x", tensor_type, list(x.shape)),
                helper.make_tensor_value_info("bias", tensor_type, list(bias.shape)),
            ],
            outputs=[helper.make_tensor_value_info("y", tensor_type, list(x.shape))],
        )

        model = helper.make_model(graph, producer_name=f"{op_name}_test")
        verify_with_ort_with_inputs(
            model,
            [x, bias],
            [x.shape],
            atol=absolute_tolerance,
            dtype=data_type,
            target=target,
            dev=dev,
        )

    x = np.array([-1.0, 0, 1.0, 100.0, -100.0, 1000.0, -1000.0], dtype=dtype)
    bias = np.repeat(2.0, 7).astype(dtype)
    verify_biasgelu(x, bias)

    x = np.array([[1, 2], [3, 4]], dtype=dtype)
    bias = np.array([0.3, 4.0], dtype=dtype)
    verify_biasgelu(x, bias)


@tvm.testing.parametrize_targets
def test_embedlayernormalization(target, dev):
    """test_embedlayernormalization"""

    def verify_embedlayernormalization(
        input_ids,
        segment_ids,
        word_embedding,
        position_embedding,
        segment_embedding,
        gamma,
        beta,
    ):
        node = onnx.helper.make_node(
            "EmbedLayerNormalization",
            inputs=[
                "input_ids",
                "" if segment_ids is None else "segment_ids",
                "word_embedding",
                "position_embedding",
                "" if segment_embedding is None else "segment_embedding",
                "gamma",
                "beta",
            ],
            outputs=["output", "mask_index"],
            domain="com.microsoft",
        )

        node.attribute.append(onnx.helper.make_attribute("epsilon", 1e-4))

        segment_ids_shape = [] if segment_ids is None else segment_ids.shape
        segment_embedding_shape = [] if segment_embedding is None else segment_embedding.shape

        graph = helper.make_graph(
            [node],
            "embedlayernormalization_test",
            inputs=[
                helper.make_tensor_value_info(
                    "input_ids", TensorProto.INT32, list(input_ids.shape)
                ),
                helper.make_tensor_value_info("segment_ids", TensorProto.INT32, segment_ids_shape),
                helper.make_tensor_value_info(
                    "word_embedding", TensorProto.FLOAT, list(word_embedding.shape)
                ),
                helper.make_tensor_value_info(
                    "position_embedding", TensorProto.FLOAT, list(position_embedding.shape)
                ),
                helper.make_tensor_value_info(
                    "segment_embedding", TensorProto.FLOAT, segment_embedding_shape
                ),
                helper.make_tensor_value_info("gamma", TensorProto.FLOAT, list(gamma.shape)),
                helper.make_tensor_value_info("beta", TensorProto.FLOAT, list(beta.shape)),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, list((batch_size, sequence_length, hidden_size))
                ),
                helper.make_tensor_value_info("mask_index", TensorProto.INT32, [batch_size]),
            ],
        )

        model = helper.make_model(graph, producer_name="embedlayernormalization_test")

        # TODO(@anwang2009): onnxruntime v1.9.0 requires empty list for optional argument,
        # but v1.10.0+ requires None instead.
        verify_with_ort_with_inputs(
            model,
            [
                input_ids,
                np.empty(0, dtype="int32") if segment_ids is None else segment_ids,
                word_embedding,
                position_embedding,
                np.empty(0, dtype="float32") if segment_embedding is None else segment_embedding,
                gamma,
                beta,
            ],
            [
                (batch_size, sequence_length, hidden_size),
                batch_size,
            ],
            target=target,
            dev=dev,
            rtol=1e-4,
            atol=1e-4,
        )

    hidden_size = 384
    batch_size = 4
    sequence_length = 3
    vocab_size = 5

    input_ids = np.full((batch_size, sequence_length), 3).astype("int32")
    segment_ids = np.zeros((batch_size, sequence_length)).astype("int32")
    word_embedding = np.full((vocab_size, hidden_size), 1).astype("float32")
    position_embedding = np.full((sequence_length, hidden_size), 2).astype("float32")
    segment_embedding = np.full((vocab_size, hidden_size), 3).astype("float32")

    gamma = np.random.uniform(0.5, 0.7, hidden_size).astype("float32")
    beta = np.random.randn(hidden_size).astype("float32") * 0.1

    verify_embedlayernormalization(
        input_ids, segment_ids, word_embedding, position_embedding, segment_embedding, gamma, beta
    )

    # Test with undefined segment embedding
    verify_embedlayernormalization(
        input_ids, None, word_embedding, position_embedding, None, gamma, beta
    )


@tvm.testing.parametrize_targets
def test_attention(target, dev):
    """test_attention"""

    def verify_attention(_unidirectional, _input, _weight, _bias, _mask_index=None, _past=None):
        input_names = ["input", "weight", "bias"]
        if _mask_index is not None:
            input_names.append("mask_index")
        if _past is not None:
            input_names.append("past")

        node = onnx.helper.make_node(
            "Attention",
            inputs=input_names,
            outputs=["output", "present"],
            domain="com.microsoft",
            num_heads=num_heads,
            unidirectional=_unidirectional,
        )

        past_shape = (2, batch_size, num_heads, past_sequence_length, head_size)
        present_output_shape = (2, batch_size, num_heads, sequence_length, head_size)

        inputs_info = [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, list(_input.shape)),
            helper.make_tensor_value_info("weight", TensorProto.FLOAT, list(_weight.shape)),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, list(_bias.shape)),
        ]
        if _mask_index is not None:
            inputs_info.append(
                helper.make_tensor_value_info(
                    "mask_index", TensorProto.INT32, list(_mask_index.shape)
                ),
            )
        if _past is not None:
            inputs_info.append(
                helper.make_tensor_value_info("past", TensorProto.FLOAT, list(past_shape))
            )

        graph = helper.make_graph(
            [node],
            "attention_test",
            inputs=inputs_info,
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, list(_input.shape)),
                helper.make_tensor_value_info(
                    "present", TensorProto.FLOAT, list(present_output_shape)
                ),
            ],
        )

        model = helper.make_model(graph, producer_name="attention_test")

        inputs = [_input, _weight, _bias]
        if _mask_index is not None:
            inputs.append(_mask_index)
        if _past is not None:
            inputs.append(_past)

        # "present" output should be nullptr when the "past" input isn't included,
        # but ort requires an output shape to be specified?
        verify_with_ort_with_inputs(
            model,
            inputs,
            [_input.shape, present_output_shape],
            target=target,
            dev=dev,
            rtol=1e-4,
            atol=1e-4,
        )

    batch_size = 11
    num_heads = 13
    head_size = 37
    sequence_length = 7
    input_hidden_size = 147
    weight_hidden_size = num_heads * head_size
    past_sequence_length = 17

    total_sequence_length = past_sequence_length + sequence_length

    # Required inputs
    input_array = np.random.normal(size=(batch_size, sequence_length, input_hidden_size)).astype(
        "float32"
    )
    weight = (
        np.random.normal(size=(input_hidden_size, 3 * weight_hidden_size)).astype("float32") * 0.1
    )
    bias = np.random.randn(3 * weight_hidden_size).astype("float32")

    # Optional inputs
    past = np.random.random((2, batch_size, num_heads, past_sequence_length, head_size)).astype(
        "float32"
    )

    for unidirectional in [0, 1]:
        for have_past in [False, True]:
            if not have_past:
                mask_index = np.random.randint(0, 2, (batch_size, sequence_length)).astype("int32")
                verify_attention(unidirectional, input_array, weight, bias, mask_index)
            else:
                mask_index = np.random.randint(0, 2, (batch_size, total_sequence_length)).astype(
                    "int32"
                )
                verify_attention(unidirectional, input_array, weight, bias, mask_index, past)


@tvm.testing.parametrize_targets
def test_qattention(target, dev):
    """test_qattention"""

    def verify_attention(
        _unidirectional,
        _input,
        _weight,
        _bias,
        _input_scale,
        _weight_scale,
        _mask_index=None,
        _input_zero_point=None,
        _weight_zero_point=None,
        _past=None,
    ):
        input_names = ["input", "weight", "bias", "input_scale", "weight_scale"]
        if _mask_index is not None:
            input_names.append("mask_index")
        if _input_zero_point is not None:
            input_names.append("input_zero_point")
        if _weight_zero_point is not None:
            input_names.append("weight_zero_point")
        if _past is not None:
            input_names.append("past")

        node = onnx.helper.make_node(
            "QAttention",
            inputs=input_names,
            outputs=["output", "present"],
            domain="com.microsoft",
            num_heads=num_heads,
            unidirectional=_unidirectional,
        )

        past_shape = (2, batch_size, num_heads, past_sequence_length, head_size)
        present_output_shape = (
            2,
            batch_size,
            num_heads,
            past_sequence_length + sequence_length,
            head_size,
        )

        inputs_info = [
            helper.make_tensor_value_info("input", TensorProto.UINT8, list(_input.shape)),
            helper.make_tensor_value_info("weight", TensorProto.UINT8, list(_weight.shape)),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, list(_bias.shape)),
            helper.make_tensor_value_info("input_scale", TensorProto.FLOAT, ()),
            helper.make_tensor_value_info("weight_scale", TensorProto.FLOAT, ()),
        ]
        if _mask_index is not None:
            inputs_info.append(
                helper.make_tensor_value_info(
                    "mask_index", TensorProto.INT32, list(_mask_index.shape)
                )
            )
        if _input_zero_point is not None:
            inputs_info.append(
                helper.make_tensor_value_info("input_zero_point", TensorProto.UINT8, ())
            )
        if _weight_zero_point is not None:
            inputs_info.append(
                helper.make_tensor_value_info("weight_zero_point", TensorProto.UINT8, ())
            )
        if _past is not None:
            inputs_info.append(
                helper.make_tensor_value_info("past", TensorProto.FLOAT, list(past_shape))
            )

        graph = helper.make_graph(
            [node],
            "qattention_test",
            inputs=inputs_info,
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, list(_input.shape)),
                helper.make_tensor_value_info(
                    "present", TensorProto.FLOAT, list(present_output_shape)
                ),
            ],
        )

        model = helper.make_model(graph, producer_name="qattention_test")

        inputs = [_input, _weight, _bias, _input_scale, _weight_scale]
        if _mask_index is not None:
            inputs.append(_mask_index)
        if _input_zero_point is not None:
            inputs.append(_input_zero_point)
        if _weight_zero_point is not None:
            inputs.append(_weight_zero_point)
        if _past is not None:
            inputs.append(_past)

        verify_with_ort_with_inputs(
            model,
            inputs,
            [_input.shape, present_output_shape],
            target=target,
            dev=dev,
            rtol=1e-3,
            atol=1e-3,
        )

    batch_size = 11
    num_heads = 13
    head_size = 37
    sequence_length = 7
    input_hidden_size = 147
    weight_hidden_size = num_heads * head_size
    past_sequence_length = 17

    total_sequence_length = past_sequence_length + sequence_length

    # Required inputs
    input_array = np.random.randint(
        0, 255, (batch_size, sequence_length, input_hidden_size)
    ).astype("uint8")
    weight = np.random.randint(0, 255, (input_hidden_size, 3 * weight_hidden_size)).astype("uint8")
    bias = np.random.randn(3 * weight_hidden_size).astype("float32")
    input_scale = np.random.random(1).astype("float32")
    weight_scale = np.random.random(1).astype("float32")

    # Optional inputs
    input_zero_point = np.random.randint(0, 255, 1).astype("uint8")
    weight_zero_point = np.random.randint(0, 255, 1).astype("uint8")
    past = np.random.random((2, batch_size, num_heads, past_sequence_length, head_size)).astype(
        "float32"
    )

    for unidirectional in [0, 1]:
        for have_past in [False, True]:
            if not have_past:
                mask_index = np.random.randint(0, 2, (batch_size, sequence_length)).astype("int32")

                verify_attention(
                    unidirectional,
                    input_array,
                    weight,
                    bias,
                    input_scale,
                    weight_scale,
                    mask_index,
                )
                verify_attention(
                    unidirectional,
                    input_array,
                    weight,
                    bias,
                    input_scale,
                    weight_scale,
                    mask_index,
                    input_zero_point,
                )
                verify_attention(
                    unidirectional,
                    input_array,
                    weight,
                    bias,
                    input_scale,
                    weight_scale,
                    mask_index,
                    input_zero_point,
                    weight_zero_point,
                )
            else:
                mask_index = np.random.randint(0, 2, (batch_size, total_sequence_length)).astype(
                    "int32"
                )

                verify_attention(
                    unidirectional,
                    input_array,
                    weight,
                    bias,
                    input_scale,
                    weight_scale,
                    mask_index,
                    input_zero_point,
                    weight_zero_point,
                    past,
                )


@tvm.testing.parametrize_targets
def test_skiplayernormalization(target, dev):
    """test_skiplayernormalization"""

    def verify_skiplayernormalization(input_, skip, gamma, beta, bias):
        node = onnx.helper.make_node(
            "SkipLayerNormalization",
            inputs=["input", "skip", "gamma", "beta", "bias"],
            outputs=["output"],
            domain="com.microsoft",
        )

        node.attribute.append(onnx.helper.make_attribute("epsilon", 1e-4))

        graph = helper.make_graph(
            [node],
            "skiplayernormalization_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_.shape)),
                helper.make_tensor_value_info("skip", TensorProto.FLOAT, list(skip.shape)),
                helper.make_tensor_value_info("gamma", TensorProto.FLOAT, list(gamma.shape)),
                helper.make_tensor_value_info("beta", TensorProto.FLOAT, list(beta.shape)),
                helper.make_tensor_value_info("bias", TensorProto.FLOAT, list(bias.shape)),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, list(input_.shape)),
            ],
        )

        model = helper.make_model(graph, producer_name="skiplayernormalization_test")
        verify_with_ort_with_inputs(
            model, [input_, skip, gamma, beta, bias], [input_.shape], target=target, dev=dev
        )

    hidden_size = 384
    batch_size = 4
    sequence_length = 4

    dtype = "float32"
    input_array = np.random.random((batch_size, sequence_length, hidden_size)).astype(dtype)
    skip = np.random.random((batch_size, sequence_length, hidden_size)).astype(dtype)
    gamma = np.random.uniform(0.5, 0.7, hidden_size).astype(dtype)
    beta = np.random.randn(hidden_size).astype(dtype) * 0.1
    bias = np.random.randn(hidden_size).astype(dtype)

    verify_skiplayernormalization(input_array, skip, gamma, beta, bias)


@tvm.testing.known_failing_targets("cuda")
@tvm.testing.parametrize_targets
def test_qgemm(target, dev):
    """test_qgemm"""

    def verify_qgemm(
        a_shape,
        b_shape,
        y_shape,
        C=False,
        y_zp=False,
        b_per_tensor_quantization=False,
        alpha=1.0,
        transA=0,
        transB=1,
    ):
        a_array = np.random.randint(low=0, high=255, size=a_shape).astype("uint8")
        b_array = np.random.uniform(low=0, high=255, size=b_shape).astype("uint8")

        input_nodes = [
            helper.make_tensor_value_info("a", TensorProto.UINT8, list(a_shape)),
            helper.make_tensor_value_info("b", TensorProto.UINT8, list(b_shape)),
        ]

        initializer = [
            helper.make_tensor("a_scale", TensorProto.FLOAT, (), [np.random.rand()]),
            helper.make_tensor("a_zero_point", TensorProto.UINT8, (), [np.random.randint(0, 255)]),
        ]

        input_names = [
            "a",
            "a_scale",
            "a_zero_point",
            "b",
            "b_scale",
            "b_zero_point",
        ]
        input_values = [a_array, b_array]

        if b_per_tensor_quantization:
            initializer.append(
                helper.make_tensor("b_scale", TensorProto.FLOAT, (), [np.random.rand()])
            )
            initializer.append(
                helper.make_tensor(
                    "b_zero_point", TensorProto.UINT8, (), [np.random.randint(0, 255)]
                )
            )
        else:  # per_colume_quantization
            shape_value = b_shape[0] if transB else b_shape[1]
            b_scale_array = np.random.random(shape_value).astype("float32")
            w_zero_point_array = np.random.randint(0, 255, size=shape_value).astype("uint8")
            initializer.append(
                helper.make_tensor(
                    "b_scale", TensorProto.FLOAT, list(b_scale_array.shape), b_scale_array
                )
            )
            initializer.append(
                helper.make_tensor(
                    "b_zero_point",
                    TensorProto.UINT8,
                    list(w_zero_point_array.shape),
                    w_zero_point_array,
                )
            )

        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(y_shape))

        if C is True:
            C_shape = (b_shape[0] if transB else b_shape[1],)
            C_array = np.random.randint(low=0, high=65536, size=C_shape).astype("int32")
            input_nodes.append(helper.make_tensor_value_info("C", TensorProto.INT32, list(C_shape)))
            input_names.append("C")
            input_values.append(C_array)

        if y_zp is True:
            input_names.append("y_scale")
            initializer.append(
                helper.make_tensor("y_scale", TensorProto.FLOAT, (), [np.random.rand()])
            )

            input_names.append("y_zero_point")
            initializer.append(
                helper.make_tensor(
                    "y_zero_point", TensorProto.UINT8, (), [np.random.randint(0, 255)]
                )
            )

            output_tensor = helper.make_tensor_value_info(
                "output", TensorProto.UINT8, list(y_shape)
            )

        kwargs = {}
        kwargs["alpha"] = alpha
        kwargs["transA"] = transA
        kwargs["transB"] = transB

        node = helper.make_node(
            "QGemm",
            inputs=input_names,
            outputs=["output"],
            domain="com.microsoft",
            # Default values for other attributes:
            **kwargs,
        )

        graph = helper.make_graph(
            [node],
            "QGemm",
            inputs=input_nodes,
            outputs=[output_tensor],
            initializer=initializer,
        )
        model = helper.make_model(
            graph,
            producer_name="QGemm",
            opset_imports=[
                onnx.helper.make_opsetid("com.microsoft", 1),
            ],
        )

        verify_with_ort_with_inputs(model, input_values, target=target, dev=dev)

    # B per tensor quantization
    verify_qgemm(
        (20, 30),
        (50, 30),
        (20, 50),
        True,
        True,
        True,
    )

    # B per column  quantization
    verify_qgemm(
        (20, 30),
        (50, 30),
        (20, 50),
        True,
        True,
        False,
    )

    # test alpha
    verify_qgemm(
        (20, 30),
        (50, 30),
        (20, 50),
        True,
        True,
        True,
        0.5,
    )

    # test transpose A
    verify_qgemm(
        (20, 50),
        (20, 80),
        (50, 80),
        True,
        True,
        True,
        0.5,
        1,
        0,
    )


@tvm.testing.known_failing_targets("cuda")
@tvm.testing.parametrize_targets
def test_qlinearconv(target, dev):
    """test_qlinearconv"""

    def verify_qlinearconv(
        x_shape,
        w_shape,
        y_shape,
        padding,
        kernel_shape,
        strides,
        dilations,
        auto_pad="NOTSET",
        bias=False,
        per_channel_quantization=False,
    ):

        x_array = np.random.randint(low=0, high=255, size=x_shape).astype("uint8")
        w_array = np.random.uniform(low=0, high=255, size=w_shape).astype("uint8")

        initializer = [
            helper.make_tensor("x_scale", TensorProto.FLOAT, (), [np.random.rand()]),
            helper.make_tensor("x_zero_point", TensorProto.UINT8, (), [np.random.randint(0, 255)]),
            helper.make_tensor("y_scale", TensorProto.FLOAT, (), [np.random.rand()]),
            helper.make_tensor("y_zero_point", TensorProto.UINT8, (), [np.random.randint(0, 255)]),
        ]

        input_nodes = [
            helper.make_tensor_value_info("x", TensorProto.UINT8, list(x_shape)),
            helper.make_tensor_value_info("w", TensorProto.UINT8, list(w_shape)),
        ]
        input_names = [
            "x",
            "x_scale",
            "x_zero_point",
            "w",
            "w_scale",
            "w_zero_point",
            "y_scale",
            "y_zero_point",
        ]
        input_values = [x_array, w_array]

        if per_channel_quantization:
            w_scale_array = np.random.random(w_shape[0]).astype("float32")
            w_zero_point_array = np.random.randint(0, 255, size=w_shape[0]).astype("uint8")

            initializer.append(
                helper.make_tensor("w_scale", TensorProto.FLOAT, [w_shape[0]], w_scale_array)
            )
            initializer.append(
                helper.make_tensor(
                    "w_zero_point", TensorProto.UINT8, [w_shape[0]], w_zero_point_array
                )
            )
        else:
            initializer.append(
                helper.make_tensor("w_scale", TensorProto.FLOAT, (), [np.random.rand()])
            )
            initializer.append(
                helper.make_tensor(
                    "w_zero_point", TensorProto.UINT8, (), [np.random.randint(0, 255)]
                )
            )

        if bias is True:
            b_shape = w_shape[0:1]
            b_array = np.random.randint(low=0, high=65536, size=b_shape).astype("int32")
            input_nodes.append(helper.make_tensor_value_info("B", TensorProto.INT32, list(b_shape)))
            input_names.append("B")
            input_values.append(b_array)

        if padding is None:
            ## autopadding with unset default attributes
            kwargs = {}
            if not all(list(s == 1 for s in strides)):
                kwargs["strides"] = strides
            if not all(list(d == 1 for d in dilations)):
                kwargs["dilations"] = dilations

            node = helper.make_node(
                "QLinearConv",
                inputs=input_names,
                outputs=["y"],
                # Default values for other attributes:
                auto_pad=auto_pad,
                **kwargs,
            )
        else:
            node = helper.make_node(
                "QLinearConv",
                inputs=input_names,
                outputs=["y"],
                kernel_shape=kernel_shape,
                # Default values for other attributes:
                strides=strides,
                dilations=dilations,
                # groups=1
                pads=padding,
            )

        graph = helper.make_graph(
            [node],
            "conv_test",
            inputs=input_nodes,
            outputs=[helper.make_tensor_value_info("y", TensorProto.UINT8, list(y_shape))],
            initializer=initializer,
        )
        model = helper.make_model(graph, producer_name="qlinearconv_test")
        # opt_level=1 will cause error
        verify_with_ort_with_inputs(model, input_values, opt_level=2, target=target, dev=dev)

    def repeat(num, dims):
        return tuple(num for _ in range(dims))

    # only support QLinearConv2d because only support qnn.conv2d
    dims = 2

    # Convolution with padding
    verify_qlinearconv(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(5, dims),
        2 * repeat(1, dims),
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
    )

    # Convolution with bias
    verify_qlinearconv(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(5, dims),
        2 * repeat(1, dims),
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
        bias=True,
    )

    # Convolution with asymmetric padding
    verify_qlinearconv(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(4, dims),
        repeat(0, dims) + repeat(1, dims),
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
    )
    # Convolution without padding
    verify_qlinearconv(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(3, dims),
        2 * repeat(0, dims),
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
    )
    # Convolution with autopadding
    verify_qlinearconv(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(5, dims),
        None,
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
        auto_pad="SAME_UPPER",
    )
    # Convolution with valid autopadding
    verify_qlinearconv(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(3, dims),
        None,
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
        auto_pad="VALID",
    )
    # Convolution with non uniform stride
    verify_qlinearconv(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(3, dims),
        None,
        repeat(3, dims),
        repeat(2, dims),
        repeat(1, dims),
        auto_pad="SAME_UPPER",
    )
    # Convolution with dilation
    verify_qlinearconv(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(5, dims),
        2 * repeat(2, dims),
        repeat(3, dims),
        repeat(1, dims),
        repeat(2, dims),
    )
    # Convolution with per channel quantization
    verify_qlinearconv(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(3, dims),
        None,
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
        per_channel_quantization=True,
    )


# TODO(vvchernov): fix problem with quantization on cuda
@tvm.testing.known_failing_targets("cuda")
@tvm.testing.parametrize_targets
def test_qlinearmatmul(target, dev):
    """test_qlinearmatmul"""

    def verify_qlinearmatmul(
        x_shape,
        w_shape,
        y_shape,
        x_dtype="uint8",
        w_dtype="uint8",
    ):
        def get_randint_numpy_scalar(dtype="uint8"):
            if dtype == "uint8":
                return np.random.randint(0, 255)
            else:  # "int8"
                return np.random.randint(-128, 127)

        if x_dtype == "uint8":
            x_array = np.random.randint(low=0, high=255, size=x_shape).astype("uint8")
        else:  # "int8"
            x_array = np.random.randint(low=-128, high=127, size=x_shape).astype("int8")
        if w_dtype == "uint8":
            w_array = np.random.uniform(low=0, high=255, size=w_shape).astype("uint8")
        else:  # "int8"
            w_array = np.random.uniform(low=-128, high=127, size=w_shape).astype("int8")

        x_proto_type = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(x_dtype)]
        w_proto_type = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(w_dtype)]

        y_dtype = "int8"
        if x_dtype == "uint8" and w_dtype == "uint8":
            y_dtype = "uint8"
        y_proto_type = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(y_dtype)]

        initializer = [
            helper.make_tensor("x_scale", TensorProto.FLOAT, (), [np.random.rand()]),
            # TODO: 0 value for int8?
            helper.make_tensor(
                "x_zero_point", x_proto_type, (), [get_randint_numpy_scalar(x_dtype)]
            ),
            helper.make_tensor("w_scale", TensorProto.FLOAT, (), [np.random.rand()]),
            # TODO: 0 value for int8?
            helper.make_tensor(
                "w_zero_point", w_proto_type, (), [get_randint_numpy_scalar(w_dtype)]
            ),
            helper.make_tensor("y_scale", TensorProto.FLOAT, (), [np.random.rand()]),
            helper.make_tensor(
                "y_zero_point", y_proto_type, (), [get_randint_numpy_scalar(y_dtype)]
            ),
        ]

        input_nodes = [
            helper.make_tensor_value_info("x", x_proto_type, list(x_shape)),
            helper.make_tensor_value_info("w", w_proto_type, list(w_shape)),
        ]
        input_names = [
            "x",
            "x_scale",
            "x_zero_point",
            "w",
            "w_scale",
            "w_zero_point",
            "y_scale",
            "y_zero_point",
        ]
        input_values = [x_array, w_array]

        node = helper.make_node(
            "QLinearMatMul",
            inputs=input_names,
            outputs=["y"],
        )

        y_proto_type = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype("int8")]
        if x_dtype == "uint8" and w_dtype == "uint8":
            y_proto_type = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype("uint8")]

        graph = helper.make_graph(
            [node],
            "qmatmul_test",
            inputs=input_nodes,
            outputs=[helper.make_tensor_value_info("y", y_proto_type, list(y_shape))],
            initializer=initializer,
        )
        model = helper.make_model(graph, producer_name="qlinearmatmul_test")
        # opt_level=1 will cause error
        verify_with_ort_with_inputs(model, input_values, opt_level=2, target=target, dev=dev)

    # Default matmul both ranks = 2 (x_dtype = "uint8", w_dtype = "uint8")
    verify_qlinearmatmul((2, 3), (3, 2), (2, 2))

    # Default matmul both ranks = 2 (x_dtype = "int8", w_dtype = "int8")
    verify_qlinearmatmul((2, 3), (3, 2), (2, 2), "int8", "int8")

    # TODO(vvchernov): problems on ONNX Runtime side and type check (onnx.py:L4763) on TVM side
    # Default matmul both ranks = 2 (x_dtype = "uint8", w_dtype = "int8")
    # verify_qlinearmatmul((2, 3), (3, 2), (2, 2), "uint8", "int8")

    # TODO(vvchernov): problems on ONNX Runtime side and type check (onnx.py:L4763) on TVM side
    # Default matmul both ranks = 2 (x_dtype = "int8", w_dtype = "uint8")
    # verify_qlinearmatmul((2, 3), (3, 2), (2, 2), "int8", "uint8")

    # Reduced matmul: x_ranks = 1, w_rank = 2 (x_dtype = "uint8", w_dtype = "uint8")
    verify_qlinearmatmul((3,), (3, 2), (2,))

    # Special case matmul: x_ranks = 3, w_rank = 2 (x_dtype = "uint8", w_dtype = "uint8")
    verify_qlinearmatmul((2, 3, 4), (4, 3), (2, 3, 3))

    # GPT2-style matmul both ranks = 4 (x_dtype = "uint8", w_dtype = "uint8")
    verify_qlinearmatmul((2, 4, 3, 3), (2, 4, 3, 3), (2, 4, 3, 3))

    # Asymetric matmul: x_ranks = 4, w_rank = 3 (x_dtype = "uint8", w_dtype = "uint8")
    verify_qlinearmatmul((2, 4, 3, 3), (4, 3, 3), (2, 4, 3, 3))

    # Asymetric matmul: x_ranks = 2, w_rank = 3 (x_dtype = "uint8", w_dtype = "uint8")
    # verify_qlinearmatmul((3, 3), (4, 3, 3), (4, 3, 3))

@tvm.testing.parametrize_targets("llvm")
def test_random_bernoulli(target, dev):
    """test_random_bernoulli"""

    def _get_tvm_output(
        inputs,
        out_dtype="int32",
        seed=None,
        target=target,
        dev=dev,
        use_vm=False,
        freeze_params=False,
    ):
        def get_bernoulli_model(shape, in_dtype="float32", out_dtype="int32", seed=None):
            onnx_itype = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(in_dtype)]
            onnx_otype = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(out_dtype)]
            node = helper.make_node(
                "Bernoulli",
                ["input"],
                ["output"],
            )
            dtype_attr = helper.make_attribute("dtype", onnx_otype)
            node.attribute.append(dtype_attr)
            if seed is not None:
                seed_attr = helper.make_attribute("seed", float(seed))
                node.attribute.append(seed_attr)

            graph = helper.make_graph(
                [node],
                "random_bernoulli_test",
                inputs=[helper.make_tensor_value_info("input", onnx_itype, list(shape))],
                outputs=[helper.make_tensor_value_info("output", onnx_otype, list(shape))],
            )
            return helper.make_model(graph, producer_name="random_bernoulli_test")

        shape = inputs.shape
        in_dtype = inputs.dtype
        model = get_bernoulli_model(shape, in_dtype, out_dtype, seed)

        if use_vm:
            return get_tvm_output_with_vm(
                model,
                inputs,
                target,
                dev,
                freeze_params=freeze_params,
            )
        else:
            return get_tvm_output(
                model,
                inputs,
                target,
                dev,
            )

    def binom_test(input, ideal_mean, threshold=0.05):
        # This test is strictly appropriate when input probabilities are all identical.
        # In that case, it should lead to flaky failures in only one run in a million (p>=1e-6).
        # The test should be over-conservative when input probabilities are not identical.
        # (i.e., It should have a rate of flaky failures lower than one run in a million.)
        # If this test starts repeatedly throwing flaky failures, consult a statistician
        # in addition to your regular debugging.
        bnm_test_res = scipy.stats.binomtest(
            k=np.sum(input, dtype="int32"), n=len(input), p=ideal_mean
        )
        return bnm_test_res.pvalue > threshold

    def verify_bernoulli(
        inputs=None,
        shape=[],
        in_dtype="float32",
        out_dtype="int32",
        seed=None,
        target=target,
        dev=dev,
        use_vm=False,
        freeze_params=False,
        in_out_equal=False,
    ):
        if inputs is None:
            assert len(shape) != 0
            inputs = np.random.uniform(size=shape).astype(in_dtype)

        tvm_out = _get_tvm_output(
            inputs,
            out_dtype,
            seed,
            target,
            dev,
            use_vm,
            freeze_params,
        )

        if isinstance(tvm_out, list):
            tvm_out = tvm_out[0]
        # check that values are 0 or 1
        tvm_flat = tvm_out.flatten()
        assert np.array_equal(tvm_flat, tvm_flat.astype("bool"))
        if in_out_equal:
            tvm.testing.assert_allclose(inputs, tvm_out)
        else:
            # check that mean value is close to the theoretical one by binomial test
            ideal_mean = np.mean(inputs)
            repeats = 3
            check = False
            for i in range(repeats):
                if binom_test(tvm_flat, ideal_mean):
                    check = True
                    break
                else:
                    # repeat with new seed
                    seed = np.random.randint(1e6)
                    tvm_flat = _get_tvm_output(
                        inputs,
                        out_dtype,
                        seed,
                        target,
                        dev,
                        use_vm,
                        freeze_params,
                    ).flatten()
            assert check, "Binomial test failed"

    # Test input sequence of 0 and 1
    inputs = np.random.randint(2, size=[10000]).astype("float32")
    verify_bernoulli(inputs, in_out_equal=True)

    # Binomial test input with 0.5 values
    val_num = 10000
    inputs = np.ones([val_num], dtype="float32") * 0.5
    verify_bernoulli(inputs)

    # Binomial test input with 0.1 values
    inputs = np.ones([val_num], dtype="float32") * 0.1
    verify_bernoulli(inputs)

    # Simple test
    verify_bernoulli(shape=[val_num])

    # Floating output type
    verify_bernoulli(shape=[val_num], out_dtype="float32")

    # Double input type
    verify_bernoulli(shape=[val_num], in_dtype="float64")

    # Test N-D tensor generation
    verify_bernoulli(shape=[2, 4, 100, 100])

    # Test with seed
    verify_bernoulli(shape=[val_num], seed=np.random.randint(1e6))

    # Test result determinism with the same seeds
    inputs = np.random.uniform(size=[val_num])
    fixed_seed = np.random.randint(1e6)
    tvm_out_1 = _get_tvm_output(inputs, seed=fixed_seed)
    tvm_out_2 = _get_tvm_output(inputs, seed=fixed_seed)
    tvm.testing.assert_allclose(tvm_out_1, tvm_out_2)


@tvm.testing.parametrize_targets("llvm")
def test_random_uniform(target, dev):
    """test_random_uniform"""

    def get_random_uniform(shape, dtype="float32", high=1.0, low=0.0, seed=None):
        ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
        node = helper.make_node(
            "RandomUniform", [], ["out"], shape=shape, dtype=ONNX_DTYPE, high=high, low=low
        )
        if seed is not None:
            seed_attr = helper.make_attribute("seed", seed)
            node.attribute.append(seed_attr)

        graph = helper.make_graph(
            [node],
            "random_uniform_test",
            inputs=[],
            outputs=[helper.make_tensor_value_info("out", ONNX_DTYPE, shape)],
        )
        model = helper.make_model(graph, producer_name="random_uniform_test")
        return get_tvm_output_with_vm(
            model,
            [],
            target=target,
            dev=dev,
            validate_structural_equal=(seed is not None),
        )

    # Check that function runs and produces proper shape.
    vals = get_random_uniform([10], dtype="float32")
    assert list(vals.shape) == [10]
    assert vals.dtype == "float32"

    # Test N-D tensor generation.
    vals = get_random_uniform([1, 3, 100, 100], dtype="float32")
    assert list(vals.shape) == [1, 3, 100, 100]

    # Check that bounds aren't exceeded.
    vals = get_random_uniform(shape=[100], high=100.0, low=-100.0)
    assert list(vals.shape) == [100]
    assert all(vals >= -100) and all(vals <= 100)

    # Check that a fixed seed produces the same values when run twice.
    vals_1 = get_random_uniform(shape=[10], seed=1)
    vals_2 = get_random_uniform(shape=[10], seed=1)
    assert all(vals_1 == vals_2)

    # Test against an expected output with a fixed seed.
    real = get_random_uniform(shape=[10], seed=5.0)
    expected = np.asarray(
        [
            0.043976,
            0.96656,
            0.292199,
            0.904297,
            0.25167,
            0.521778,
            0.778985,
            0.085463,
            0.939846,
            0.194201,
        ]
    )
    tvm.testing.assert_allclose(real, expected, rtol=1e-5)


@tvm.testing.parametrize_targets("llvm")
def test_random_uniform_like(target, dev):
    """test_random_uniform_like"""

    def get_random_uniform_like(input_, shape, dtype=None, high=1.0, low=0.0, seed=None):
        node = helper.make_node("RandomUniformLike", ["in"], ["out"], high=high, low=low)
        if seed is not None:
            seed_attr = helper.make_attribute("seed", seed)
            node.attribute.append(seed_attr)

        ONNX_DTYPE = None
        if dtype is not None:
            ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
            dtype_attr = helper.make_attribute("dtype", ONNX_DTYPE)
            node.attribute.append(dtype_attr)
        else:
            dtype = input_.dtype
            ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]

        graph = helper.make_graph(
            [node],
            "random_uniform_test",
            inputs=[helper.make_tensor_value_info("in", ONNX_DTYPE, shape)],
            outputs=[helper.make_tensor_value_info("out", ONNX_DTYPE, shape)],
        )
        model = helper.make_model(graph, producer_name="random_uniform_like_test")
        return get_tvm_output_with_vm(
            model,
            [input_],
            target=target,
            dev=dev,
            validate_structural_equal=(seed is not None),
        )

    # Check that function runs and produces proper shape and dtype.
    shape = [10]
    input_array = np.random.random(shape).astype("float32")
    vals = get_random_uniform_like(input_array, shape, dtype="float32")
    assert list(vals.shape) == [10]
    assert vals.dtype == "float32"

    # Test N-D tensor generation.
    shape = [1, 3, 100, 100]
    input_array = np.random.random(shape).astype("float32")
    vals = get_random_uniform_like(input_array, shape, dtype="float64")
    assert list(vals.shape) == shape
    assert vals.dtype == "float64"

    # Check that bounds aren't exceeded.
    shape = [100]
    input_array = np.random.random(shape).astype("float64")
    vals = get_random_uniform_like(input_array, shape, high=100.0, low=-100.0)
    assert list(vals.shape) == shape
    assert all(vals >= -100) and all(vals <= 100)

    # Test against an expected output with a fixed seed.
    shape = [10]
    input_array = np.random.random(shape).astype("float32")
    real = get_random_uniform_like(input_array, shape=[10], seed=5.0)
    expected = np.asarray(
        [
            0.043976,
            0.96656,
            0.292199,
            0.904297,
            0.25167,
            0.521778,
            0.778985,
            0.085463,
            0.939846,
            0.194201,
        ]
    )
    tvm.testing.assert_allclose(real, expected, rtol=1e-5)


@tvm.testing.parametrize_targets("llvm")
def test_random_normal(target, dev):
    """test_random_normal"""

    def get_random_normal(shape, dtype="float32", scale=1.0, mean=0.0, seed=None):
        ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
        node = helper.make_node(
            "RandomNormal", [], ["out"], shape=shape, dtype=ONNX_DTYPE, scale=scale, mean=mean
        )
        if seed is not None:
            seed_attr = helper.make_attribute("seed", seed)
            node.attribute.append(seed_attr)

        graph = helper.make_graph(
            [node],
            "random_normal_test",
            inputs=[],
            outputs=[helper.make_tensor_value_info("out", ONNX_DTYPE, shape)],
        )
        model = helper.make_model(graph, producer_name="random_normal_test")
        return get_tvm_output_with_vm(
            model,
            [],
            target=target,
            dev=dev,
            validate_structural_equal=(seed is not None),
        )

    # Test N-D tensor generation.
    vals = get_random_normal([1, 3, 100, 100], dtype="float32")
    assert list(vals.shape) == [1, 3, 100, 100]
    tvm.testing.assert_allclose(vals.mean(), 0.0, rtol=0.1, atol=0.1)
    tvm.testing.assert_allclose(np.std(vals), 1.0, rtol=0.1, atol=0.1)

    # Test mean=2.0 scale=10.0
    vals = get_random_normal([1, 3, 100, 100], mean=2.0, scale=10.0, dtype="float32")
    assert list(vals.shape) == [1, 3, 100, 100]
    tvm.testing.assert_allclose(vals.mean(), 2.0, rtol=0.1, atol=0.1)
    tvm.testing.assert_allclose(np.std(vals), 10.0, rtol=0.1, atol=0.1)

    # Check that a fixed seed produces the same values when run twice.
    vals_1 = get_random_normal(shape=[10], seed=1.0)
    vals_2 = get_random_normal(shape=[10], seed=1.0)
    assert all(vals_1 == vals_2)


@tvm.testing.parametrize_targets("llvm")
def test_random_normal_like(target, dev):
    """test_random_normal_like"""

    def get_random_normal_like(input_, shape, dtype="float32", scale=1.0, mean=0.0, seed=None):
        ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
        node = helper.make_node(
            "RandomNormalLike", ["in"], ["out"], dtype=ONNX_DTYPE, scale=scale, mean=mean
        )
        if seed is not None:
            seed_attr = helper.make_attribute("seed", seed)
            node.attribute.append(seed_attr)

        graph = helper.make_graph(
            [node],
            "random_normal_like_test",
            inputs=[helper.make_tensor_value_info("in", ONNX_DTYPE, shape)],
            outputs=[helper.make_tensor_value_info("out", ONNX_DTYPE, shape)],
        )
        model = helper.make_model(graph, producer_name="random_normal_like_test")
        return get_tvm_output_with_vm(
            model,
            [input_],
            target=target,
            dev=dev,
            validate_structural_equal=(seed is not None),
        )

    # Test N-D tensor generation.
    shape = [1, 3, 100, 100]
    input_array = np.random.random(shape).astype("float32")
    vals = get_random_normal_like(input_array, [1, 3, 100, 100], dtype="float32")
    assert list(vals.shape) == [1, 3, 100, 100]
    tvm.testing.assert_allclose(vals.mean(), 0.0, rtol=0.1, atol=0.1)
    tvm.testing.assert_allclose(np.std(vals), 1.0, rtol=0.1, atol=0.1)

    # Test mean=2.0 scale=10.0
    shape = [1, 3, 100, 100]
    input_array = np.random.random(shape).astype("float32")
    vals = get_random_normal_like(
        input_array, [1, 3, 100, 100], mean=2.0, scale=10.0, dtype="float32"
    )
    assert list(vals.shape) == [1, 3, 100, 100]
    tvm.testing.assert_allclose(vals.mean(), 2.0, rtol=0.1, atol=0.1)
    tvm.testing.assert_allclose(np.std(vals), 10.0, rtol=0.1, atol=0.1)


@tvm.testing.parametrize_targets("llvm")
def test_multinomial(target, dev):
    def get_multinomial(input, shape, sample_size, seed=None):
        IN_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype("float32")]
        OUT_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype("int32")]
        node = helper.make_node("Multinomial", ["in"], ["out"], sample_size=sample_size)
        if seed is not None:
            seed_attr = helper.make_attribute("seed", seed)
            node.attribute.append(seed_attr)

        graph = helper.make_graph(
            [node],
            "multinomial_test",
            inputs=[helper.make_tensor_value_info("in", IN_DTYPE, shape)],
            outputs=[helper.make_tensor_value_info("out", OUT_DTYPE, shape)],
        )
        model = helper.make_model(graph, producer_name="multinomial_test")
        return get_tvm_output_with_vm(
            model,
            [input],
            target=target,
            dev=dev,
            validate_structural_equal=(seed is not None),
        )

    # Test N-D tensor generation.
    shape = [3]
    sample_size = 2
    probs = np.random.random(shape).astype("float32")
    indices = get_multinomial(probs, shape, sample_size)
    # Since specific values are random, we'll check that the output shape is
    # correct and the values chosen are all valid indices.
    assert list(indices.shape) == [sample_size]
    assert np.max(indices) < shape[-1]

    # Test 2d multinomial
    shape = [10, 5]
    sample_size = 4
    probs = np.random.random(shape).astype("float32")
    indices = get_multinomial(probs, shape, sample_size)
    assert list(indices.shape) == [10, sample_size]
    assert np.max(indices) < shape[-1]


@tvm.testing.parametrize_targets
def test_convinteger(target, dev):
    """test_convinteger"""

    def verify_convinteger(
        x_shape,
        w_shape,
        y_shape,
        padding,
        kernel_shape,
        strides,
        dilations,
        auto_pad="NOTSET",
        dtype="uint8",
    ):
        x_array = np.random.randint(low=0, high=255, size=x_shape).astype(dtype)
        w_array = np.random.uniform(low=0, high=255, size=w_shape).astype(dtype)
        x_zero_point_array = np.random.randint(0, 255, size=[1]).astype(dtype)
        w_zero_point_array = np.random.randint(0, 255, size=[1]).astype(dtype)

        ONNX_DTYPE = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
        input_nodes = [
            helper.make_tensor_value_info("x", ONNX_DTYPE, list(x_shape)),
            helper.make_tensor_value_info("w", ONNX_DTYPE, list(w_shape)),
        ]
        initializer = [
            helper.make_tensor("x_zero_point", ONNX_DTYPE, [], x_zero_point_array),
            helper.make_tensor("w_zero_point", ONNX_DTYPE, [], w_zero_point_array),
        ]
        input_names = ["x", "w", "x_zero_point", "w_zero_point"]
        input_values = [x_array, w_array]

        if padding is None:
            ## autopadding with unset default attributes
            kwargs = {}
            if not all(list(s == 1 for s in strides)):
                kwargs["strides"] = strides
            if not all(list(d == 1 for d in dilations)):
                kwargs["dilations"] = dilations

            node = helper.make_node(
                "ConvInteger",
                inputs=input_names,
                outputs=["y"],
                # Default values for other attributes:
                auto_pad=auto_pad,
                **kwargs,
            )
        else:
            node = helper.make_node(
                "ConvInteger",
                inputs=input_names,
                outputs=["y"],
                kernel_shape=kernel_shape,
                # Default values for other attributes:
                strides=strides,
                dilations=dilations,
                # groups=1
                pads=padding,
            )

        graph = helper.make_graph(
            [node],
            "convinteger_test",
            inputs=input_nodes,
            initializer=initializer,
            outputs=[helper.make_tensor_value_info("y", TensorProto.INT32, list(y_shape))],
        )
        model = helper.make_model(graph, producer_name="convinteger_test")
        # opt_level=1 will cause error
        verify_with_ort_with_inputs(model, input_values, target=target, dev=dev, opt_level=2)

    def repeat(num, dims):
        return tuple(num for _ in range(dims))

    # only support 2D ConvInteger because we only support qnn.conv2d for now.
    dims = 2

    # Convolution with padding
    verify_convinteger(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(5, dims),
        2 * repeat(1, dims),
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
    )

    # Convolution with asymmetric padding
    verify_convinteger(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(4, dims),
        repeat(0, dims) + repeat(1, dims),
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
    )
    # Convolution without padding
    verify_convinteger(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(3, dims),
        2 * repeat(0, dims),
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
    )
    # Convolution with autopadding
    verify_convinteger(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(5, dims),
        None,
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
        auto_pad="SAME_UPPER",
    )
    # Convolution with valid autopadding
    verify_convinteger(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(3, dims),
        None,
        repeat(3, dims),
        repeat(1, dims),
        repeat(1, dims),
        auto_pad="VALID",
    )
    # Convolution with non uniform stride
    verify_convinteger(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(3, dims),
        None,
        repeat(3, dims),
        repeat(2, dims),
        repeat(1, dims),
        auto_pad="SAME_UPPER",
    )
    # Convolution with dilation
    verify_convinteger(
        (1, 1) + repeat(5, dims),
        (1, 1) + repeat(3, dims),
        (1, 1) + repeat(5, dims),
        2 * repeat(2, dims),
        repeat(3, dims),
        repeat(1, dims),
        repeat(2, dims),
    )


@tvm.testing.parametrize_targets
def test_bitshift(target, dev):
    """test_bitshift"""

    def verify_bitshift(in_shape, shift_shape, high=1000000000, in_dtype="uint64"):
        in_shape = list(in_shape)
        shift_shape = list(shift_shape)

        # Create an input for each tensor.
        tensor_values = [
            np.random.randint(high, size=in_shape).astype(in_dtype),
            np.random.randint(16, size=shift_shape).astype(in_dtype),
            np.random.randint(16, size=shift_shape).astype(in_dtype),
        ]

        bitshift_left_node = helper.make_node(
            "BitShift",
            inputs=["input", "shift_left"],
            outputs=["shifted"],
            direction="LEFT",
        )

        bitshift_right_node = helper.make_node(
            "BitShift",
            inputs=["shifted", "shift_right"],
            outputs=["output"],
            direction="RIGHT",
        )

        # Create input and output tensors.
        proto_type = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(in_dtype)]
        graph_inputs = [
            helper.make_tensor_value_info("input", proto_type, in_shape),
            helper.make_tensor_value_info("shift_left", proto_type, shift_shape),
            helper.make_tensor_value_info("shift_right", proto_type, shift_shape),
        ]

        graph_outputs = [helper.make_tensor_value_info("output", proto_type, in_shape)]

        graph_nodes = [bitshift_left_node, bitshift_right_node]

        graph = helper.make_graph(
            graph_nodes,
            "BitShift_test",
            inputs=graph_inputs,
            outputs=graph_outputs,
        )
        model = helper.make_model(
            graph,
            producer_name="BitShift_test",
        )

        verify_with_ort_with_inputs(model, tensor_values, target=target, dev=dev)

    shape = (100, 4, 2)
    broadcast_shape = (100, 1, 1)
    # Common bitwise test
    verify_bitshift(shape, shape)
    # Bitwise test with broadcasting
    verify_bitshift(shape, broadcast_shape)


# TODO(vvchernov): return test back than ONNX Runtime in CI will support domain version of 18
@pytest.mark.skip("Currently ONNX Runtime in CI does not support domain version of 18")
@tvm.testing.parametrize_targets
def test_bitwise(target, dev):
    """test_bitwise"""

    def verify_bitwise_ops(A_shape, B_shape, C_shape, D_shape, high=128, in_dtype="int32"):
        A_shape = list(A_shape)
        B_shape = list(B_shape)
        C_shape = list(C_shape)
        D_shape = list(D_shape)

        # Create an input for each tensor.
        tensor_values = [
            np.random.randint(high, size=A_shape).astype(in_dtype),
            np.random.randint(high, size=B_shape).astype(in_dtype),
            np.random.randint(high, size=C_shape).astype(in_dtype),
            np.random.randint(high, size=D_shape).astype(in_dtype),
        ]

        or_node = helper.make_node(
            "BitwiseOr",
            inputs=["A", "B"],
            outputs=["OR"],
        )

        and_node = helper.make_node(
            "BitwiseAnd",
            inputs=["OR", "C"],
            outputs=["AND"],
        )

        xor_node = helper.make_node(
            "BitwiseXor",
            inputs=["AND", "D"],
            outputs=["XOR"],
        )

        not_node = helper.make_node(
            "BitwiseNot",
            inputs=["XOR"],
            outputs=["output"],
        )

        # Create input and output tensors.
        proto_type = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(in_dtype)]
        graph_inputs = [
            helper.make_tensor_value_info("A", proto_type, A_shape),
            helper.make_tensor_value_info("B", proto_type, B_shape),
            helper.make_tensor_value_info("C", proto_type, C_shape),
            helper.make_tensor_value_info("D", proto_type, D_shape),
        ]

        graph_outputs = [
            helper.make_tensor_value_info("output", proto_type, A_shape),
        ]

        graph_nodes = [
            or_node,
            and_node,
            xor_node,
            not_node,
        ]

        graph = helper.make_graph(
            graph_nodes,
            "Bitwise_test",
            inputs=graph_inputs,
            outputs=graph_outputs,
        )
        model = helper.make_model(
            graph,
            producer_name="Bitwise_test",
        )

        verify_with_ort_with_inputs(model, tensor_values, target=target, dev=dev)

    shape = (100, 4, 2)
    broadcast_shape = (100, 1, 1)
    dtypes = ["int8", "uint8", "int32", "uint32"]
    high_vals = [128, 128, 2147483648, 2147483648]
    for high, dtype in zip(high_vals, dtypes):
        # Common bitwise test
        verify_bitwise_ops(shape, shape, shape, shape, high, dtype)
        # Bitwise test with broadcasting
        verify_bitwise_ops(shape, broadcast_shape, broadcast_shape, broadcast_shape, high, dtype)


@tvm.testing.parametrize_targets
def test_scan(target, dev):
    """test_scan"""

    def verify_scan(
        input_shapes,
        output_shapes,
        num_scan_inputs,
        scan_input_axes,
        scan_input_directions,
        scan_output_axes,
        scan_output_directions,
        opset,
    ):

        body_input_shapes = copy.deepcopy(input_shapes)
        num_state_inputs = len(input_shapes) - num_scan_inputs

        if opset == 8:
            for i in range(len(input_shapes)):
                body_input_shapes[i].pop(0)
            for i in range(num_state_inputs, len(input_shapes)):
                body_input_shapes[i].pop(0)
        else:
            for i in range(num_state_inputs, len(input_shapes)):
                body_input_shapes[i].pop(scan_input_axes[i - num_state_inputs])

        initial0 = onnx.helper.make_tensor_value_info(
            "initial0", onnx.TensorProto.FLOAT, body_input_shapes[0]
        )
        initial1 = onnx.helper.make_tensor_value_info(
            "initial1", onnx.TensorProto.FLOAT, body_input_shapes[1]
        )
        input0 = onnx.helper.make_tensor_value_info(
            "input0", onnx.TensorProto.FLOAT, body_input_shapes[2]
        )
        input1 = onnx.helper.make_tensor_value_info(
            "input1", onnx.TensorProto.FLOAT, body_input_shapes[3]
        )
        input2 = onnx.helper.make_tensor_value_info(
            "input2", onnx.TensorProto.FLOAT, body_input_shapes[4]
        )
        state0 = onnx.helper.make_tensor_value_info(
            "state0", onnx.TensorProto.FLOAT, body_input_shapes[0]
        )
        scan_out0 = onnx.helper.make_tensor_value_info(
            "scan_out0", onnx.TensorProto.FLOAT, body_input_shapes[0]
        )
        state1 = onnx.helper.make_tensor_value_info(
            "state1", onnx.TensorProto.FLOAT, body_input_shapes[1]
        )
        scan_out1 = onnx.helper.make_tensor_value_info(
            "scan_out1", onnx.TensorProto.FLOAT, body_input_shapes[1]
        )
        add_node = onnx.helper.make_node(
            "Add",
            inputs=["initial0", "input0"],
            outputs=["state0"],
        )
        id_node_0 = onnx.helper.make_node(
            "Identity",
            inputs=["state0"],
            outputs=["scan_out0"],
        )
        matmul_node = onnx.helper.make_node(
            "MatMul",
            inputs=["input1", "input2"],
            outputs=["matmul_out"],
        )
        sub_node = onnx.helper.make_node(
            "Sub",
            inputs=["initial1", "matmul_out"],
            outputs=["state1"],
        )
        id_node_1 = onnx.helper.make_node(
            "Identity",
            inputs=["state1"],
            outputs=["scan_out1"],
        )
        scan_body = onnx.helper.make_graph(
            [add_node, id_node_0, matmul_node, sub_node, id_node_1],
            "scan_body",
            [initial0, initial1, input0, input1, input2],
            [state0, state1, scan_out0, scan_out1],
        )
        # create scan op node
        scan_node = None
        if opset == 8:
            scan_node = onnx.helper.make_node(
                "Scan",
                inputs=["", "init0", "init1", "in0", "in1", "in2"],
                outputs=["s0", "s1", "scan0", "scan1"],
                num_scan_inputs=num_scan_inputs,
                body=scan_body,
            )
        else:
            scan_node = onnx.helper.make_node(
                "Scan",
                inputs=["init0", "init1", "in0", "in1", "in2"],
                outputs=["s0", "s1", "scan0", "scan1"],
                num_scan_inputs=num_scan_inputs,
                body=scan_body,
                scan_input_axes=scan_input_axes,
                scan_input_directions=scan_input_directions,
                scan_output_axes=scan_output_axes,
                scan_output_directions=scan_output_directions,
            )
        input_info = [
            helper.make_tensor_value_info("init0", TensorProto.FLOAT, input_shapes[0]),
            helper.make_tensor_value_info("init1", TensorProto.FLOAT, input_shapes[1]),
            helper.make_tensor_value_info("in0", TensorProto.FLOAT, input_shapes[2]),
            helper.make_tensor_value_info("in1", TensorProto.FLOAT, input_shapes[3]),
            helper.make_tensor_value_info("in2", TensorProto.FLOAT, input_shapes[4]),
        ]
        out_info = [
            helper.make_tensor_value_info("s0", TensorProto.FLOAT, output_shapes[0]),
            helper.make_tensor_value_info("s1", TensorProto.FLOAT, output_shapes[1]),
            helper.make_tensor_value_info("scan0", TensorProto.FLOAT, output_shapes[2]),
            helper.make_tensor_value_info("scan1", TensorProto.FLOAT, output_shapes[3]),
        ]
        graph = helper.make_graph(
            nodes=[scan_node],
            name="scan_test",
            inputs=input_info,
            outputs=out_info,
        )
        model = onnx.helper.make_model(graph, producer_name="scan-test")
        init0 = np.random.uniform(low=0, high=255, size=input_shapes[0]).astype(np.float32)
        init1 = np.random.uniform(low=0, high=255, size=input_shapes[1]).astype(np.float32)
        in0 = np.random.uniform(low=0, high=255, size=input_shapes[2]).astype(np.float32)
        in1 = np.random.uniform(low=0, high=255, size=input_shapes[3]).astype(np.float32)
        in2 = np.random.uniform(low=0, high=255, size=input_shapes[4]).astype(np.float32)
        input_values = [init0, init1, in0, in1, in2]

        verify_with_ort_with_inputs(
            model,
            input_values,
            target=target,
            dev=dev,
            opt_level=2,
            use_vm=True,
            opset=opset,
        )

    # opset 8
    input_shapes = [[2, 6, 7, 8], [2, 3, 3], [2, 5, 6, 7, 8], [2, 5, 3, 4], [2, 5, 4, 3]]
    output_shapes = [[2, 6, 7, 8], [2, 3, 3], [2, 5, 6, 7, 8], [2, 5, 3, 3]]
    # input_shapes, output_shapes, num_scan_inputs, scan_input_axes, scan_input_directions,
    # scan_output_axes, scan_output_directions, opset
    verify_scan(input_shapes, output_shapes, 3, [0] * 3, [0] * 3, [0] * 2, [0] * 2, 8)
    # opset 9
    input_shapes = [[6, 7, 8], [3, 3], [5, 6, 7, 8], [5, 3, 4], [5, 4, 3]]
    output_shapes = [[6, 7, 8], [3, 3], [5, 6, 7, 8], [5, 3, 3]]
    verify_scan(input_shapes, output_shapes, 3, [0] * 3, [0] * 3, [0] * 2, [0] * 2, 9)

    input_shapes = [[6, 7, 8], [3, 3], [5, 6, 7, 8], [3, 4, 5], [4, 5, 3]]
    output_shapes = [[6, 7, 8], [3, 3], [6, 5, 7, 8], [3, 5, 3]]
    verify_scan(input_shapes, output_shapes, 3, [0, 2, 1], [1] * 3, [1] * 2, [1] * 2, 9)
    # Negative axes
    input_shapes = [[6, 7, 8], [3, 3], [5, 6, 7, 8], [3, 4, 5], [4, 5, 3]]
    output_shapes = [[6, 7, 8], [3, 3], [6, 5, 7, 8], [3, 5, 3]]
    verify_scan(input_shapes, output_shapes, 3, [-4, -1, -2], [1] * 3, [-3, -2], [1] * 2, 9)


@tvm.testing.parametrize_targets
def test_linear_regressor(target, dev):
    """test_linear_regressor"""

    def verify_linear_regressor(a_shape, c_shape, i_shape, targets=1, batch=1):
        a_array = np.random.uniform(size=a_shape).astype("float32")
        out_shape = (batch, targets)

        coefficients = np.random.uniform(size=c_shape).astype("float32")
        intercepts = np.random.uniform(size=i_shape).astype("float32")

        mul_node = helper.make_node(
            "LinearRegressor",
            ["a"],
            ["out"],
            coefficients=coefficients,
            intercepts=intercepts,
            targets=targets,
            domain="ai.onnx.ml",
        )

        graph = helper.make_graph(
            [mul_node],
            "LinearRegressor_test",
            inputs=[
                helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)],
        )
        model = helper.make_model(
            graph,
            producer_name="LinearRegressor_test",
            opset_imports=[
                onnx.helper.make_opsetid("ai.onnx.ml", 1),
            ],
        )
        verify_with_ort_with_inputs(model, [a_array], target=target, dev=dev)

    verify_linear_regressor((1, 3), (3), (1))
    verify_linear_regressor((2, 10), (10), (1), batch=2)
    verify_linear_regressor((1, 3), (30), (10), targets=10)
    verify_linear_regressor((10, 3), (30), (10), targets=10, batch=10)
    verify_linear_regressor((1, 4), (3), (1))


@tvm.testing.parametrize_targets
def test_dft(target, dev):
    """test_dft"""

    def verify_dft(
        _axis,
        _inverse,
        _onesided,
        _dft_length,
        _input_shape,
        _output_shape,
    ):
        input_names = ["input"]
        if _dft_length is not None:
            input_names.append("dft_length")

        node = onnx.helper.make_node(
            "DFT",
            inputs=input_names,
            outputs=["output"],
            axis=_axis,
            inverse=_inverse,
            onesided=_onesided,
        )

        nodes = []
        if _dft_length is not None:
            nodes.append(
                make_constant_node("dft_length", TensorProto.INT32, [], [_dft_length]),
            )
        nodes.append(node)

        graph = helper.make_graph(
            nodes,
            "dft_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, _input_shape),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, _output_shape),
            ],
        )

        model = helper.make_model(graph, producer_name="dft_test")

        _input = np.random.normal(size=_input_shape).astype("float32")
        verify_with_ort_with_inputs(
            model,
            [_input],
            [_input_shape],
            target=target,
            dev=dev,
            rtol=1e-4,
            atol=1e-4,
            use_vm=False,
        )

    batch_size = 5
    n = 2
    D = 7

    for axis in list(range(1, n)) + [-2]:
        for inverse, onesided in [(0, 0), (0, 1), (1, 0)]:
            for n_fft in [D, D - 1, D + 1]:
                for c in [1, 2]:
                    input_shape = [batch_size] + n * [D] + [c]
                    output_shape = [batch_size] + n * [D] + [2]
                    if onesided == 1:
                        output_shape[axis] = output_shape[axis] // 2 + 1
                    verify_dft(axis, inverse, onesided, n_fft, input_shape, output_shape)


@tvm.testing.parametrize_targets
def test_sequence(target, dev):
    """test_sequence"""

    def verify_sequence_ops(tensor_shape, num_tensors, axis=0, position=0, new_axis=None):
        tensor_shape = list(tensor_shape)
        tensor_values = []
        for i in range(num_tensors):
            tensor_values.append(np.random.uniform(size=tensor_shape).astype("float32"))

        # Create an input for each tensor.
        input_tensor_names = []
        for i in range(num_tensors):
            name = f"input_tensor_{i}"
            input_tensor_names.append(name)

        # Test creating a tensor sequence.
        construct_node = helper.make_node(
            "SequenceConstruct",
            inputs=input_tensor_names,
            outputs=["sequence"],
        )

        position_node = make_constant_node("position", TensorProto.INT32, (), [position])

        # Test sequence insertion.
        insert_node = helper.make_node(
            "SequenceInsert",
            inputs=["sequence", input_tensor_names[0], "position"],
            outputs=["inserted_sequence"],
        )

        # Test sequence erase.
        erase_node = helper.make_node(
            "SequenceErase",
            inputs=["inserted_sequence", "position"],
            outputs=["erased_sequence"],
        )

        # Test sequence concatenation.
        concat_node = helper.make_node(
            "ConcatFromSequence",
            inputs=["erased_sequence"],
            outputs=["concat_sequence"],
            axis=axis,
        )

        # Test splitting a tensor into a sequence.
        split_node = helper.make_node(
            "SplitToSequence", inputs=["concat_sequence"], outputs=["split_sequence"], axis=axis
        )

        # Test tensor extraction from sequence
        at_node = helper.make_node(
            "SequenceAt", inputs=["split_sequence", "position"], outputs=["output"]
        )

        # Test sequence length
        length_node = helper.make_node(
            "SequenceLength", inputs=["split_sequence"], outputs=["output_2"]
        )

        if new_axis is not None:
            new_axis_attr = helper.make_attribute("new_axis", new_axis)
            concat_node.attribute.append(new_axis_attr)

        # Create input and output tensors.
        graph_inputs = []
        for name in input_tensor_names:
            input_tensor = helper.make_tensor_value_info(name, TensorProto.FLOAT, tensor_shape)
            graph_inputs.append(input_tensor)

        # Construct output tensor.
        output_shape = tensor_shape
        if new_axis is not None:
            output_shape.insert(axis, 1)
            output_shape[axis] = num_tensors + 1
        else:
            output_shape[axis] = (num_tensors + 1) * output_shape[axis]
        graph_outputs = [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape),
            helper.make_tensor_value_info("output_2", TensorProto.INT64, []),
        ]

        graph_nodes = [
            position_node,
            construct_node,
            insert_node,
            erase_node,
            concat_node,
            split_node,
            at_node,
            length_node,
        ]

        graph = helper.make_graph(
            graph_nodes,
            "Sequence_test",
            inputs=graph_inputs,
            outputs=graph_outputs,
        )
        model = helper.make_model(
            graph,
            producer_name="Sequence_test",
        )

        verify_with_ort_with_inputs(model, tensor_values, target=target, dev=dev)

    verify_sequence_ops((10, 3), 2)
    verify_sequence_ops((3, 3, 3, 3), 4, position=3)
    verify_sequence_ops((3, 3, 3, 3), 4, axis=2)
    verify_sequence_ops((3, 3, 3, 3), 4, axis=2, new_axis=1)


@tvm.testing.parametrize_targets
def test_empty_sequence(target, dev):
    """test_empty_sequence"""

    # Test creating an empty tensor sequence.
    empty_node = helper.make_node(
        "SequenceEmpty",
        inputs=[],
        outputs=["empty_sequence"],
    )

    length_node = helper.make_node("SequenceLength", inputs=["empty_sequence"], outputs=["output"])

    graph_outputs = [helper.make_tensor_value_info("output", TensorProto.INT64, [])]

    graph_nodes = [empty_node, length_node]

    graph = helper.make_graph(
        graph_nodes,
        "Sequence_empty_test",
        inputs=[],
        outputs=graph_outputs,
    )

    model = helper.make_model(
        graph,
        producer_name="Sequence_empty_test",
    )

    verify_with_ort_with_inputs(model, [], target=target, dev=dev)


def test_exporting_node_renamed_model():
    """test exproting model when export_node_renamed_model is set"""

    a_name, a_shape = "a", (4, 3)
    b_name, b_shape = "b", (3, 4)
    out_name, out_shape = "out", [a_shape[0], b_shape[1]]
    temp_dir = utils.tempdir().path

    # model definition
    mul_node = helper.make_node("MatMul", [a_name, b_name], [out_name])
    graph = helper.make_graph(
        [mul_node],
        "matmul_test",
        inputs=[
            helper.make_tensor_value_info(a_name, TensorProto.FLOAT, a_shape),
            helper.make_tensor_value_info(b_name, TensorProto.FLOAT, b_shape),
        ],
        outputs=[helper.make_tensor_value_info(out_name, TensorProto.FLOAT, out_shape)],
    )
    model = helper.make_model(graph, producer_name="matmul_test")

    # get frontend model
    shape_dict = {a_name: a_shape, b_name: b_shape}
    _, _ = relay.frontend.from_onnx(model, shape_dict, export_node_renamed_model_path=temp_dir)

    exported_model_name = os.listdir(temp_dir)[0]
    assert "tvm_exported_model_" in exported_model_name

    exported_model = onnx.load(os.path.join(temp_dir, exported_model_name))
    assert exported_model.graph.node[0].name == "MatMul_0"


class TestSetSpan:
    """test structural equal between translated / hand-crafted relay IR with span tagged."""

    def _verify(self, res_fptr, golden_fptr):
        with tvm.testing.enable_span_filling():
            with_span = res_fptr()
        with tvm.testing.disable_span_filling():
            without_span = res_fptr()
        assert tvm.ir.structural_equal(with_span, without_span)
        _verify_structural_equal_with_span(with_span, golden_fptr())

    def test_conv2d_bias_add_span(self):
        padding = [0, 0, 0, 0]
        k_shape = [7, 7]
        y_shape, y_name = [1, 6, 10, 10], "y"
        x_shape, x_name = [1, 3, 10, 10], "x"
        b_shape, b_name = [6], "b"
        b_val = np.random.random(b_shape).astype(np.float32)
        w_shape, w_name = [6, 3, 7, 7], "w"
        w_val = np.random.random(w_shape).astype(np.float32)
        group, strides, dilations = 1, [1, 1], [1, 1]
        conv_name = "conv2d"

        def _res():
            # model definition
            node = helper.make_node(
                "Conv",
                inputs=[x_name, w_name, b_name],
                outputs=[y_name],
                kernel_shape=k_shape,
                strides=strides,
                dilations=dilations,
                group=group,
                pads=padding,
                name=conv_name,
            )
            graph = helper.make_graph(
                [node],
                "conv_test",
                inputs=[helper.make_tensor_value_info(x_name, TensorProto.FLOAT, x_shape)],
                outputs=[helper.make_tensor_value_info(y_name, TensorProto.FLOAT, y_shape)],
                initializer=[
                    helper.make_tensor(
                        w_name,
                        TensorProto.FLOAT,
                        dims=w_shape,
                        vals=w_val.flatten(),
                    ),
                    helper.make_tensor(
                        b_name,
                        TensorProto.FLOAT,
                        dims=b_shape,
                        vals=b_val.flatten(),
                    ),
                ],
            )
            model = helper.make_model(graph, producer_name="conv_test")

            # get frontend model
            shape_dict = {x_name: x_shape}
            mod, _ = relay.frontend.from_onnx(model, shape_dict)
            return mod["main"]

        def _golden():
            conv_si = conv_name
            x = relay.var(
                x_name,
                shape=tuple(x_shape),
                span=_create_span(f"{conv_si}.{x_name}"),
            )
            conv_weight = relay.const(
                w_val,
                span=_create_span(f"{conv_si}.{w_name}"),
            )
            conv_bias = relay.const(
                b_val,
                span=_create_span(f"{conv_si}.{b_name}"),
            )
            conv_out = _set_span(
                relay.nn.conv2d(
                    x,
                    conv_weight,
                    padding=[0] * 4,
                    channels=y_shape[1],
                    kernel_size=k_shape,
                ),
                conv_si,
            )
            bias_out = _set_span(relay.nn.bias_add(conv_out, conv_bias), conv_si)
            return infer_type(relay.Function([x], bias_out))

        self._verify(_res, _golden)

    def test_batchnorm_span(self):
        input_name, in_shape = "x", [1, 16, 10, 10]
        bn_name = "bn"
        output_name = "y"
        scale_name = "scale"
        bias_name = "b"
        mean_name = "mean"
        var_name = "var"

        def _res():
            # model definition
            batchnorm = onnx.helper.make_node(
                "BatchNormalization",
                inputs=[input_name, scale_name, bias_name, mean_name, var_name],
                outputs=[output_name],
                name=bn_name,
            )
            graph = helper.make_graph(
                [batchnorm],
                "batchnorm_test",
                inputs=[
                    helper.make_tensor_value_info(input_name, TensorProto.FLOAT, in_shape),
                    helper.make_tensor_value_info(scale_name, TensorProto.FLOAT, [in_shape[1]]),
                    helper.make_tensor_value_info(bias_name, TensorProto.FLOAT, [in_shape[1]]),
                    helper.make_tensor_value_info(mean_name, TensorProto.FLOAT, [in_shape[1]]),
                    helper.make_tensor_value_info(var_name, TensorProto.FLOAT, [in_shape[1]]),
                ],
                outputs=[helper.make_tensor_value_info(output_name, TensorProto.FLOAT, in_shape)],
            )
            model = helper.make_model(graph, producer_name="batchnorm_test")

            # get frontend model
            shape_dict = {input_name: in_shape}
            mod, _ = relay.frontend.from_onnx(model, shape_dict)
            return mod["main"]

        def _golden():
            bn_si = bn_name
            x = relay.var(
                input_name,
                shape=tuple(in_shape),
                span=_create_span(f"{bn_si}.{input_name}"),
            )
            bn_scale = relay.var(
                scale_name,
                shape=(in_shape[1],),
                span=_create_span(f"{bn_si}.{scale_name}"),
            )
            bn_bias = relay.var(
                bias_name,
                shape=(in_shape[1],),
                span=_create_span(f"{bn_si}.{bias_name}"),
            )
            bn_rm = relay.var(
                mean_name,
                shape=(in_shape[1],),
                span=_create_span(f"{bn_si}.{mean_name}"),
            )
            bn_rv = relay.var(
                var_name,
                shape=(in_shape[1],),
                span=_create_span(f"{bn_si}.{var_name}"),
            )
            bn_out = _set_span(
                relay.nn.batch_norm(x, bn_scale, bn_bias, bn_rm, bn_rv),
                bn_si,
            )
            bn_tuple_get_item = _set_span(relay.TupleGetItem(bn_out.tuple_value, 0), bn_si)
            return infer_type(
                relay.Function([x, bn_scale, bn_bias, bn_rm, bn_rv], bn_tuple_get_item)
            )

        self._verify(_res, _golden)

    def test_reshape_span(self):
        input_shape = [2, 1, 10, 1, 10]
        new_shape = [2, 1, 10, 10]
        input_name = "in"
        output_name = "out"
        ref_name = "ref_in"
        const_name = "const"
        reshape_name = "reshape"

        def _res():
            # model definition
            ref_array = np.array(new_shape)
            ref_node = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[ref_name],
                value=helper.make_tensor(
                    name="const_tensor",
                    data_type=TensorProto.INT32,
                    dims=ref_array.shape,
                    vals=ref_array.flatten().astype(int),
                ),
                name=const_name,
            )
            reshape_node = helper.make_node(
                "Reshape",
                [input_name, ref_name],
                [output_name],
                name=reshape_name,
            )
            graph = helper.make_graph(
                [ref_node, reshape_node],
                "reshape_test",
                inputs=[helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)],
                outputs=[helper.make_tensor_value_info(output_name, TensorProto.FLOAT, new_shape)],
            )
            model = helper.make_model(graph, producer_name="reshape_test")

            # get frontend model
            shape_dict = {input_name: input_shape}
            mod, _ = relay.frontend.from_onnx(model, shape_dict)
            return mod["main"]

        def _golden():
            reshape_si = reshape_name
            x = relay.var(
                input_name,
                shape=tuple(input_shape),
                span=_create_span(f"{reshape_si}.{input_name}"),
            )
            reshape_out = _set_span(
                relay.reshape(x, newshape=new_shape),
                reshape_si,
            )
            return infer_type(relay.Function([x], reshape_out))

        self._verify(_res, _golden)

    def test_matmul_span(self):
        a_name, a_shape = "a", (4, 3)
        b_name, b_shape = "b", (3, 4)
        out_name, out_shape = "out", [a_shape[0], b_shape[1]]
        matmul_name = "matmul"

        def _res():
            # model definition
            mul_node = helper.make_node("MatMul", [a_name, b_name], [out_name], name=matmul_name)
            graph = helper.make_graph(
                [mul_node],
                "matmul_test",
                inputs=[
                    helper.make_tensor_value_info(a_name, TensorProto.FLOAT, a_shape),
                    helper.make_tensor_value_info(b_name, TensorProto.FLOAT, b_shape),
                ],
                outputs=[helper.make_tensor_value_info(out_name, TensorProto.FLOAT, out_shape)],
            )
            model = helper.make_model(graph, producer_name="matmul_test")

            # get frontend model
            shape_dict = {a_name: a_shape, b_name: b_shape}
            mod, _ = relay.frontend.from_onnx(model, shape_dict)
            return mod["main"]

        def _golden():
            matmul_si = matmul_name
            a = relay.var(
                a_name,
                shape=tuple(a_shape),
                span=_create_span(f"{matmul_si}.{a_name}"),
            )
            b = relay.var(
                b_name,
                shape=tuple(b_shape),
                span=_create_span(f"{matmul_si}.{b_name}"),
            )
            b_t = _set_span(relay.transpose(b, axes=[1, 0]), matmul_si)
            matmul_out = _set_span(
                relay.nn.dense(a, b_t, out_dtype="float32"),
                matmul_si,
            )
            return infer_type(relay.Function([a, b], matmul_out))

        self._verify(_res, _golden)


@tvm.testing.parametrize_targets
def test_pad_constant_value(target, dev):
    """test_pad_constant_value"""

    def verify_pad_constant_value(constant_value):
        tensor_shape = [1, 2, 257, 126]
        tensor_values = [np.random.uniform(size=tensor_shape).astype("float32")]
        graph_inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, tensor_shape)]
        graph_outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]
        pads = helper.make_tensor("pads", TensorProto.INT64, [8], [0, 0, 0, 2, 0, 0, 0, 0])
        pad_node = helper.make_node(
            "Pad", ["input", "pads", constant_value], ["output"], mode="constant"
        )
        graph_nodes = [pad_node]
        graph = helper.make_graph(
            graph_nodes,
            "test_pad_constant_value",
            inputs=graph_inputs,
            outputs=graph_outputs,
            initializer=[pads],
        )
        model = helper.make_model(
            graph,
            producer_name="test_pad_constant_value",
        )
        verify_with_ort_with_inputs(model, tensor_values, target=target, dev=dev)

    verify_pad_constant_value("")
