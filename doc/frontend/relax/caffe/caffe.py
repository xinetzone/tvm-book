from dataclasses import dataclass
import numpy as np
import tvm
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
from tvm import relax
from tvm import topi
from tvm.relax.testing import nn
from tvm.relax.frontend.nn import core, modules, Module, Tensor, op, spec
from tvm.relax import op as _op
from utils import Conv2D, ConvTranspose2D

def _rebuild_layers(predict_layer):
    """Rebuild caffe layer. If the caffe net include in-place layers, repalce its top
    with its name and update the bottom of other layer that is related to it.
    """
    # dict of input name that will be changed to new name
    changed_top_dict = dict()

    for pl in predict_layer:
        if pl.type == "Input":
            continue
        # if current layer has single input and output and input equals to output
        # it means that the layer does "in-place"
        if len(pl.top) == 1 and len(pl.bottom) == 1:
            if pl.top[0] == pl.bottom[0]:
                # change current layer's input firstly
                if pl.bottom[0] in changed_top_dict:
                    pl.bottom[0] = changed_top_dict[pl.bottom[0]]
                # update "change" dict
                changed_top_dict[pl.top[0]] = pl.name
                # change current layer's output to its name
                pl.top[0] = pl.name
            else:
                if pl.bottom[0] in changed_top_dict:
                    pl.bottom[0] = changed_top_dict[pl.bottom[0]]
        # if the layer does not
        else:
            for index, plt in enumerate(pl.bottom):
                if plt in changed_top_dict:
                    pl.bottom[index] = changed_top_dict[plt]

def _get_inputs_outputs(predict_layer):
    """Obtain Caffe model's inputs and outpus"""
    # model inputs / outputs
    model_inputs = list()
    model_outputs = list()

    # The bottoms of every layer can not be as outputs
    not_outputs = set()
    for pl in predict_layer:
        if pl.type == "Input":
            assert len(pl.top) == 1, "The number of Input layer's output is more than 1."
            model_inputs.append(pl.top[0])
        for i in pl.bottom:
            not_outputs.add(i)

    for pl in predict_layer:
        if len(pl.bottom) > 0:
            for t in pl.top:
                if t not in not_outputs:
                    model_outputs.append(t)
    return model_inputs, model_outputs

def _assert_struct_info(lhs_expr, rhs_expr):
    tvm.ir.assert_structural_equal(lhs_expr.struct_info, rhs_expr.struct_info)

def convert_num_to_name(original_layer):
    """将caffe模型的top, bottom名称转换为layer_name"""
    idx2name = {}
    input_name = None
    new_layer = []
    for layer in original_layer:
        # 第一层处理
        if layer.type == "Input":
            input_name = layer.name
            new_layer.append(layer)
            continue
        # 第二层
        if layer.bottom[0] == input_name:
            idx2name[layer.top[0]] = layer.name
            layer.top[0] = layer.name
            new_layer.append(layer)
            continue
        # 之后的层
        for idx1, bottom in enumerate(layer.bottom):
            if bottom in idx2name:
                layer.bottom[idx1] = idx2name[bottom]
        for idx2, top in enumerate(layer.top):
            if top in idx2name:
                layer.top[idx2] = idx2name[top]
            else:
                idx2name[layer.top[idx2]] = layer.name
                layer.top[idx2] = layer.name
        new_layer.append(layer)
    return new_layer

@dataclass
class CaffeOperatorConverter:
    """Convert from caffe model into compatible relay Function.

    Args:
        init_net : caffe_pb2.NetParameter
            caffemodel
        predict_net : caffe_pb2.NetParameter
            caffe prototxt
        shape_dict : dict of str to int list/tuple
            Input shapes of the model.
        dtype_dict : dict of str to str
            Input types of the model.
    """
    init_net: "caffe_pb2.NetParameter"
    predict_net: "caffe_pb2.NetParameter"

    def __post_init__(self):
        self.exp_tab = {}
        self.params = {}
        self.bb: relax.BlockBuilder = relax.BlockBuilder()
        old_caffe = False
        if len(self.predict_net.input) != 0:  # old caffe version
            old_caffe = True
            model_inputs = list(self.predict_net.input)
        predict_layer = self.predict_net.layer
        # 将caffe模型中的top, bottom名称转换为layer_name, 便于后续处理
        predict_layer = convert_num_to_name(predict_layer)
        # replace layer's top with its name and update other layers'bottoms
        _rebuild_layers(predict_layer)
        # obtain inputs and outputs of Net
        if old_caffe:
            _, model_outputs = _get_inputs_outputs(predict_layer)
        else:
            model_inputs, model_outputs = _get_inputs_outputs(predict_layer)

        if list(self.init_net.layer):
            init_layer = self.init_net.layer
        else:
            init_layer = self.init_net.layers
        self.init_layer_dict = {il.name: il for il in init_layer}
        self.predict_layer = predict_layer
        self.supported_op_names = [
            "BatchNorm",
            "Concat",
            "Convolution",
            "Crop",
            "Deconvolution",
            "Dropout",
            "Eltwise",
            "Embed",
            "Flatten",
            "InnerProduct",
            "Input",
            "LRN",
            "Permute",
            "Pooling",
            "Power",
            "PReLU",
            "ReLU",
            "Reshape",
            "Scale",
            "Sigmoid",
            "Slice",
            "Softmax",
            "TanH",
            "Upsample",
            "Reduction",
        ]
        self.model_inputs = model_inputs
        self.model_outputs = model_outputs

    def check_unsupported_ops(self):
        """Check unsupported Caffe ops in our converter."""
        unsupported_ops_set = set()

        include_layer = dict()
        for pl in self.predict_layer:
            if pl.type not in include_layer:
                include_layer[pl.type] = 1
            else:
                include_layer[pl.type] = include_layer[pl.type] + 1

        for pl in self.predict_layer:
            op_name = pl.type
            if op_name not in self.supported_op_names:
                unsupported_ops_set.add(op_name)

        if unsupported_ops_set:
            msg = "The following operators are not supported in frontend " "Caffe: {}"
            ops = str(list(unsupported_ops_set)).strip("[,]")
            raise tvm.error.OpNotImplemented(msg.format(ops))

    def fuse_op(self, layers):
        """Fusing the BatchNorm and Scale layer"""
        bn, scale = layers["bn"], layers["scale"]

        # bn params
        bn_weight_bias_blobs = self.init_layer_dict[bn.name].blobs
        bn_scale = np.asarray(bn_weight_bias_blobs[2].data, np.float32)
        if bn_scale:
            bn_scale = 1 / bn_scale
        bn_mean = np.asarray(bn_weight_bias_blobs[0].data, np.float32) * bn_scale
        bn_var = np.asarray(bn_weight_bias_blobs[1].data, np.float32) * bn_scale
        bn_eps = bn.batch_norm_param.eps

        # scale params
        scale_weight_bias_blobs = self.init_layer_dict[scale.name].blobs
        scale_gamma = np.asarray(scale_weight_bias_blobs[0].data, np.float32)
        scale_bias = scale.scale_param.bias_term
        if scale_bias:
            scale_beta = np.asarray(scale_weight_bias_blobs[1].data, np.float32)
        else:
            scale_beta = np.zeros(scale_gamma.shape, dtype=np.float32)

        # new params
        self.new_bn[bn.name] = [bn_mean, bn_var, bn_eps, scale_gamma, scale_beta]
        return bn

    
    def op_fuse(self):
        """fuse bn and scale"""
        new_layers = []
        temp_layers = {}
        changed_layers = {}

        for index, pl in enumerate(self.predict_layer):
            op_type = pl.type
            if op_type == "Input":
                new_layers.append(pl)
                continue
            elif op_type == "BatchNorm":
                if (index != len(self.predict_layer) - 1) and (
                    self.predict_layer[index + 1].type == "Scale"
                ):
                    temp_layers["bn"] = pl
                    continue
                else:
                    new_layers.append(pl)
                    temp_layers.clear()
            elif op_type == "Scale":
                if self.predict_layer[index - 1].type == "BatchNorm":
                    temp_layers["scale"] = pl
                else:
                    new_layers.append(pl)
                    temp_layers.clear()
            else:
                temp_layers.clear()

            if len(temp_layers) == 2:
                layer = self.fuse_op(temp_layers)
                new_layers.append(layer)
                changed_layers[temp_layers["scale"].name] = temp_layers["bn"].name

            for idx, plt in enumerate(pl.bottom):
                if plt in changed_layers:
                    pl.bottom[idx] = changed_layers[plt]

            if op_type not in ["BatchNorm", "Scale"]:
                new_layers.append(pl)

        self.predict_layer = new_layers
        self.changed_layers = changed_layers

    def parse_inputs(self, 
        shape_dict: dict[str, list],
        dtype_dict: dict[str, str]):
        inputs = {}
        for in_name in self.model_inputs:
            shape = shape_dict[in_name] if in_name in shape_dict else None
            dtype = dtype_dict[in_name] if in_name in dtype_dict else "float32"
            inputs[in_name] = nn.Placeholder(shape, dtype, name=in_name)
        return inputs
    
    def _parse_conv_params(self, pl):
        """Parse the parameters of Convolution and Deconvolution layer"""
        nonzone = lambda val, pos, dflt: val[pos] if pos < len(val) else dflt

        conv_params = pl.convolution_param

        params = dict()
        # parse kernel size
        if conv_params.kernel_h > 0 or conv_params.kernel_w > 0:
            params["kernel_size"] = (conv_params.kernel_h, conv_params.kernel_w)
        else:
            ksize_h = nonzone(conv_params.kernel_size, 0, 1)
            ksize_w = nonzone(conv_params.kernel_size, 1, ksize_h)
            params["kernel_size"] = (ksize_h, ksize_w)

        # parse padding size
        if conv_params.pad_h > 0 or conv_params.pad_w > 0:
            params["padding"] = (conv_params.pad_h, conv_params.pad_w)
        else:
            pad_h = nonzone(conv_params.pad, 0, 0)
            pad_w = nonzone(conv_params.pad, 1, pad_h)
            params["padding"] = (pad_h, pad_w)

        # parse stride size
        if conv_params.stride_h > 0 or conv_params.stride_w > 0:
            params["strides"] = (conv_params.stride_h, conv_params.stride_w)
        else:
            stride_h = nonzone(conv_params.stride, 0, 1)
            stride_w = nonzone(conv_params.stride, 1, stride_h)
            params["strides"] = (stride_h, stride_w)

        # parse dilation size
        if hasattr(conv_params, "dilation") and len(conv_params.dilation) > 0:
            dilation = " ".join(str(d) for d in conv_params.dilation)
            dilation = tuple(map(int, dilation.split(" ")))
            params["dilation"] = dilation
            if len(dilation) == 1:
                params["dilation"] = (dilation[0], dilation[0])

        params["data_layout"] = "NCHW"
        params["groups"] = conv_params.group
        params["out_channels"] = conv_params.num_output
        params["out_dtype"] = "float32"
        return params

    def convert_conv2d(self, pl):
        _param = self._parse_conv_params(pl)
        weight_bias_blobs = self.init_layer_dict[pl.name].blobs
        conv_params = pl.convolution_param
        inputs = pl.bottom
        # process weight and bias blobs
        weight, bias = None, None
        if len(weight_bias_blobs) > 1:
            weight = weight_bias_blobs[0]
            bias = weight_bias_blobs[1]
        else:
            weight = weight_bias_blobs[0]
        if weight:
            weight_value = np.asarray(weight.data, np.float32)
            weight_value = np.reshape(weight_value, weight.shape.dim)
            _param["kernel_layout"] = "OIHW"
            _param["in_channels"] = weight_value.shape[1]*_param['groups']
        else:
            raise Exception(f"No weight value of layer {pl.name} in caffemodel")
        if bias:
            bias_value = np.asarray(bias.data, np.float32)
        
        x = self.exp_tab[inputs[0]]
        x = self.get_placeholder(x)
        _params = {"weight": weight_value, "bias": bias_value}
        model = Conv2D(**_param)
        with self.bb.function(pl.name, params=[x, *model.parameters()]):
            output = model(x)
            gv = self.bb.emit_func_output(output)
        mod = self.bb.get()
        self.params.update({pl.name: _params})
        mod = relax.transform.BindParams(pl.name, _params)(mod)
        self.bb.update_func(gv, mod[gv])
        self.exp_tab[pl.name] = gv
        return self.bb.get()
    
    def convert_relu(self, pl):
        """Convert ReLU layer"""
        inputs = pl.bottom
        x = self.exp_tab[inputs[0]]
        x = self.get_placeholder(x)
        negative_slope = pl.relu_param.negative_slope
        with self.bb.function(pl.name, params=[x]):
            if negative_slope:
                output = _op.nn.leaky_relu(x, negative_slope)
            else:
                output = _op.nn.relu(x)
            self.exp_tab[pl.name] = self.bb.emit_func_output(output)
        return self.bb.get()

    def convert_pooling(self, pl):
        """Convert Pooling layer"""
        inputs = pl.bottom
        input_name = inputs[0]

        pool_params = pl.pooling_param
        pool_type_dict = ["MAX", "AVE", "STOCHASTIC"]

        params = dict()
        # parse pool type: 0: MAX, 1: AVE, 2: STOCHASTIC
        pool_type = pool_params.pool
        # parse kernel size
        if pool_params.kernel_h > 0 or pool_params.kernel_w > 0:
            params["pool_size"] = (pool_params.kernel_h, pool_params.kernel_w)
        else:
            params["pool_size"] = (pool_params.kernel_size, pool_params.kernel_size)

        # parse padding size
        if pool_params.pad_h > 0 or pool_params.pad_w > 0:
            params["padding"] = (pool_params.pad_h, pool_params.pad_w)
        else:
            params["padding"] = (pool_params.pad, pool_params.pad)

        # parse stride size
        if pool_params.stride_h > 0 or pool_params.stride_w > 0:
            params["strides"] = (pool_params.stride_h, pool_params.stride_w)
        else:
            params["strides"] = (pool_params.stride, pool_params.stride)

        params["ceil_mode"] = True
        if hasattr(pool_params, "ceil_mode"):
            params["ceil_mode"] = pool_params.ceil_mode
        elif hasattr(pool_params, "round_mode"):
            # params["ceil_mode"] = pool_params.round_mode == "CEIL" # 有问题
            params["ceil_mode"] = pool_params.round_mode == 0

        # in_expr = self.exp_tab.get_expr(input_name)
        in_expr = self.exp_tab[input_name]
        in_expr = self.get_placeholder(in_expr)
        negative_slope = pl.relu_param.negative_slope
        with self.bb.function(pl.name, params=[in_expr]):
            if pool_type_dict[pool_type] == "MAX":
                if pool_params.global_pooling:
                    out = _op.nn.global_max_pool2d(in_expr)
                else:
                    if len(pl.top) == 1:
                        out = _op.nn.max_pool2d(in_expr, **params)
                    elif len(pl.top) == 2:
                        out1 = _op.nn.max_pool2d_with_argmax(in_expr, **params)
                        out2 = _op.vision.max_pool2d_location(in_expr, **params)
                        return _expr.Tuple((out1, out2))

            elif pool_type_dict[pool_type] == "AVE":  # AVE
                if pool_params.global_pooling:
                    out = _op.nn.global_avg_pool2d(in_expr)
                else:
                    params["count_include_pad"] = True
                    out = _op.nn.avg_pool2d(in_expr, **params)
            else:
                raise tvm.error.OpNotImplemented(
                    f"Operator {pool_type_dict[pool_type]} pool is not supported for frontend Caffe."
                )
            self.exp_tab[pl.name] = self.bb.emit_func_output(out)
        return self.bb.get()

    def get_placeholder(self, x):
        # x = self.exp_tab[inputs[0]]
        if not isinstance(x, nn.Placeholder):
            struct_info = x.struct_info.ret
            x = nn.Placeholder(list(struct_info.shape), struct_info.dtype, x.name_hint)
        return x

    def convert_eltwise(self, pl):
        """Convert Eltwise layer"""
        inputs = pl.bottom
        assert len(inputs) >= 2, "input tensors length should be larger than 2"

        # gethering initial 2 input expressions
        lhs_expr = self.get_placeholder(self.exp_tab[inputs[0]])
        rhs_expr = self.get_placeholder(self.exp_tab[inputs[1]])
        _assert_struct_info(lhs_expr, rhs_expr)

        eltwise_params = pl.eltwise_param
        eltwise_type_dict = ["PROD", "SUM", "MAX"]
        eltwise_type = eltwise_params.operation
        coeff = list(eltwise_params.coeff)
        with self.bb.function(pl.name, params=[lhs_expr, rhs_expr]):
            if eltwise_type_dict[eltwise_type] == "PROD":
                out = _op.multiply(lhs_expr, rhs_expr)
                # for rest inputs
                for i in range(len(inputs) - 2):
                    extra_expr = self.exp_tab.get_expr(inputs[i + 2])
                    _assert_struct_info(out, extra_expr)
                    out = _op.multiply(out, extra_expr)
            elif eltwise_type_dict[eltwise_type] == "SUM":
                if coeff:
                    left_coeff_expr = self.exp_tab.new_const(np.asarray(coeff[0], np.float32))
                    right_coeff_expr = self.exp_tab.new_const(np.asarray(coeff[1], np.float32))
                    lhs_expr_scale = _op.multiply(lhs_expr, left_coeff_expr)
                    rhs_expr_scale = _op.multiply(rhs_expr, right_coeff_expr)
                    out = _op.add(lhs_expr_scale, rhs_expr_scale)
                else:
                    out = _op.add(lhs_expr, rhs_expr)
                # for rest inputs
                for i in range(len(inputs) - 2):
                    extra_expr = self.exp_tab.get_expr(inputs[i + 2])
                    _assert_struct_info(out, extra_expr)
                    if coeff:
                        coeff_expr = self.exp_tab.new_const(np.asarray(coeff[i + 2], np.float32))
                        extra_expr_scale = _op.multiply(extra_expr, coeff_expr)
                        out = _op.add(out, extra_expr_scale)
                    else:
                        out = _op.add(out, extra_expr)
            elif eltwise_type_dict[eltwise_type] == "MAX":
                out = _op.maximum(lhs_expr, rhs_expr)
                # for rest inputs
                for i in range(len(inputs) - 2):
                    extra_expr = self.exp_tab.get_expr(inputs[i + 2])
                    _assert_struct_info(out, extra_expr)
                    out = _op.maximum(out, extra_expr)
            else:
                raise tvm.error.OpNotImplemented(
                    f"eltwise_type {eltwise_type} is not supported for frontend Caffe."
                )
            self.exp_tab[pl.name] = self.bb.emit_func_output(out)
        return self.bb.get()
    
    def convert_deconv(self, pl):
        """Convert Deconvolution layer"""
        params = self._parse_conv_params(pl)
        weight_bias_blobs = self.init_layer_dict[pl.name].blobs
        conv_params = pl.convolution_param
        inputs = pl.bottom

        # process weight and bias blobs
        weight, bias = None, None
        if len(weight_bias_blobs) > 1:
            weight = weight_bias_blobs[0]
            bias = weight_bias_blobs[1]
        else:
            weight = weight_bias_blobs[0]
        if weight:
            if not weight.data:
                if conv_params.weight_filler:
                    _filler = conv_params.weight_filler.value
                    weight_value = np.full(weight.shape.dim, _filler, np.float32)
                else:
                    raise tvm.error.OpAttributeInvalid("At least weight_filler must be given")
            else:
                weight_value = np.asarray(weight.data, np.float32)
            weight_value = np.reshape(weight_value, weight.shape.dim)

            # weight shape IOHW
            weight_value = np.transpose(weight_value, [1, 0, 2, 3])
            if bias:
                bias_value = np.asarray(bias.data, np.float32)
            params["in_channels"] = weight_value.shape[0]*params['groups']
            params["out_channels"] = weight_value.shape[1]
        else:
            raise tvm.error.OpAttributeRequired(f"No weight value of layer {op.name} in caffemodel")
        
        x = self.exp_tab[inputs[0]]
        x = self.get_placeholder(x)
        _params = {"weight": weight_value.transpose(1, 0, 2, 3), "bias": bias_value}
        model = ConvTranspose2D(**params)
        with self.bb.function(pl.name, params=[x, *model.parameters()]):
            output = model(x)
            gv = self.bb.emit_func_output(output)
        mod = self.bb.get()
        self.params.update({pl.name: _params})
        mod = relax.transform.BindParams(pl.name, _params)(mod)
        self.bb.update_func(gv, mod[gv])
        self.exp_tab[pl.name] = gv
        return self.bb.get()

    def convert_crop(self, pl): # TODO 需要修复
        """Convert Crop layer"""
        inputs = pl.bottom
        assert len(inputs) == 2, "Need two inputs of Crop layer"
        in_expr_a = self.get_placeholder(self.exp_tab[inputs[0]])
        in_expr_b = self.get_placeholder(self.exp_tab[inputs[1]])

        # parse crop params
        crop_params = pl.crop_param
        axis = int(getattr(crop_params, "axis", 2))
        offset = list(getattr(crop_params, "offset", 0))

        # expand offset to (offset1, offset2, ...)
        in_a_shape = in_expr_a.struct_info.shape
        num_to_crop = len(in_a_shape) - axis
        if not offset:
            offset = [0] * num_to_crop
        if len(offset) == 1:
            offset = offset * num_to_crop
        elif len(offset) != num_to_crop:
            raise tvm.error.OpAttributeInvalid("No matching the number between axis and offset!")
        
        in_b_shape = in_expr_b.struct_info.shape
        slice_end = in_b_shape
        slice_start = [0] * len(in_a_shape)
        for i in range(num_to_crop):
            slice_start[i + axis] = offset[i]
        to_crop_axis = list(range(len(in_a_shape)))
        to_crop_axis = to_crop_axis[axis:]
        with self.bb.function(pl.name, params=[in_expr_a]):
            in_expr_a_stride = _op.strided_slice(in_expr_a, to_crop_axis, slice_start, slice_end)
            self.exp_tab[pl.name] = self.bb.emit_func_output(in_expr_a_stride)
        return self.bb.get()

    def convert_innerproduct(self, pl):
        """Convert InnerProduct layer"""


def test(init_net, predict_net, shape_dict, dtype_dict):
    op_converter = CaffeOperatorConverter(init_net, predict_net)
    op_converter.check_unsupported_ops()
    op_converter.op_fuse()

    self = op_converter
    inputs = self.parse_inputs(shape_dict, dtype_dict)
    self.exp_tab.update(inputs) # 更新输入信息
    for pl in self.predict_layer:
        op_type = pl.type
        if op_type == "Input":
            continue
        output_tensors = pl.top
        if op_type == "Convolution":
            mod = self.convert_conv2d(pl)
        elif op_type == "ReLU":
            mod = self.convert_relu(pl)
        elif op_type == "Pooling":
            mod = self.convert_pooling(pl)
        elif op_type == "Eltwise":
            mod = self.convert_eltwise(pl)
        elif op_type == "Deconvolution":
            mod = self.convert_deconv(pl)
        elif op_type == "Crop":
            # mod = self.convert_crop(pl)
            break
        elif op_type == "InnerProduct":
            break