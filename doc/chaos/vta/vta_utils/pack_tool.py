"""Graph Pack 工具
"""
import tvm
from tvm import relay
from tvm.relay import ExprMutator
from tvm.relay.testing import run_opt_pass
from tvm.relay.expr import Call
from tvm.relay import op
from tvm.relay.op import op as _op
from tvm.ir.op import Op
from tvm.relay.function import Function, FunctionWithFields
from vta.top.graphpack import (
    _to_shape,
    # _unpack_batch_channel,
    _channel_const_match,
    _const_shape_match,
    _weight_shape_match_transpose, # 新增
    _pack_weight,
    _pack_weight_conv2d_transpose,
    _pack_const, # 被修改
    _get_tensor_shape,
    _get_tensor_type,
)
from .vta_conv2d import ConvAttrsTransform
from .utils import _pack_batch_channel, _channel_shape_match, _weight_shape_match

class ExprGraphPack(ExprMutator):
    def __init__(self, bfactor, cfactor, weight_bits):
        super().__init__()
        self.bfactor, self.cfactor, self.weight_bits = bfactor, cfactor, weight_bits
        self.conv2d = _op.get("nn.conv2d")
        self.conv2d_transpose = _op.get("nn.conv2d_transpose")
        self.add = _op.get("add")
        self.multiply = _op.get("multiply")
        self.bias_add = op.op.get("nn.bias_add")
        self.pad = op.op.get("nn.pad")
        self.upsampling = op.op.get("nn.upsampling")
        self.reshape = op.op.get("reshape")
        # self.cast = op.op.get("cast")
        # self.nodes = []

    def visit_call(self, call):
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        
        if isinstance(new_fn, Op):
            # oshape = _get_tensor_shape(call)
            odtype = _get_tensor_type(call)
            if call.op in [self.conv2d, self.conv2d_transpose] and "float" not in odtype:
                assert 8 % self.weight_bits == 0

                data_shape = list(_to_shape(new_args[0].checked_type.shape))
                if data_shape[1] % self.cfactor != 0:
                    new_args[0], data_shape = _channel_shape_match(new_args[0], data_shape, self.cfactor)
                new_args[0] = _pack_batch_channel(new_args[0], data_shape, self.bfactor, self.cfactor)
                # if kernel_shape[0] % self.cfactor != 0 or kernel_shape[1] % self.cfactor:
                #     # pad channels 对齐 self.cfactor
                #     new_args[0] = _pad_const(new_args[0], _to_shape(new_args[0].checked_type.shape), self.bfactor, self.cfactor)
                #     new_args[1] = _pad_const(new_args[1], _to_shape(new_args[1].checked_type.shape), self.cfactor, self.cfactor)

                w_lanes = 8 // self.weight_bits
                data_layout = "NCHW%dn%dc" % (self.bfactor, self.cfactor)
                kernel_layout = "OIHW%do%di" % (self.cfactor, self.cfactor)

                # 
                kernel_shape = _to_shape(new_args[1].checked_type.shape)
                if call.op == self.conv2d:
                    new_args[1], kernel_shape = _weight_shape_match(
                        new_args[1], kernel_shape, self.cfactor
                    )
                    channels = kernel_shape[0] # 输出通道数
                    # print(kernel_shape, channels)
                    new_args[1] = _pack_weight(new_args[1], kernel_shape, self.cfactor)
                    # insert bit packing when necessary
                    if w_lanes != 1:
                        assert 8 % w_lanes == 0
                        new_args[1] = op.bitpack(new_args[1], lanes=w_lanes)
                elif call.op == self.conv2d_transpose:
                    channels = call.attrs.channels
                    new_args[1], kernel_shape, channels = _weight_shape_match_transpose(
                        new_args[1], kernel_shape, channels, self.cfactor
                    )
                    new_args[1] = _pack_weight_conv2d_transpose(new_args[1], kernel_shape, self.cfactor)
                
                if call.op == self.conv2d:
                    call = op.nn.conv2d(
                        new_args[0],
                        new_args[1],
                        strides=call.attrs.strides,
                        padding=call.attrs.padding,
                        dilation=call.attrs.dilation,
                        groups=call.attrs.groups,
                        channels=channels,
                        kernel_size=call.attrs.kernel_size,
                        data_layout=data_layout,
                        kernel_layout=kernel_layout,
                        out_dtype=call.attrs.out_dtype,
                    )
                elif call.op == self.conv2d_transpose:
                    call = op.nn.conv2d_transpose(
                        new_args[0],
                        new_args[1],
                        strides=call.attrs.strides,
                        padding=call.attrs.padding,
                        dilation=call.attrs.dilation,
                        groups=call.attrs.groups,
                        channels=channels,
                        kernel_size=call.attrs.kernel_size,
                        data_layout=data_layout,
                        kernel_layout=kernel_layout,
                        output_padding=call.attrs.output_padding,
                        out_dtype=call.attrs.out_dtype,
                    )
                
                return run_opt_pass(call, relay.transform.InferType())
            elif (
                call.op in [self.add, self.multiply] and len(new_args[1].checked_type.shape) == 3 or
                call.op == self.bias_add
            ):
                new_args[1], input_shape = _const_shape_match(new_args[1], new_args[1].checked_type.shape, self.cfactor)
                new_args[1] = run_opt_pass(new_args[1], relay.transform.InferType())
                new_args[1] = _pack_const(
                    new_args[1], _to_shape(input_shape), new_args[1].checked_type.dtype, self.bfactor, self.cfactor
                )
            elif call.op == self.upsampling:
                (data,) = new_args
                scale_h = call.attrs.scale_h
                scale_w = call.attrs.scale_w
                data_layout = "NCHW%dn%dc" % (self.bfactor, self.cfactor)
                method = call.attrs.method
                align_corners = call.attrs.align_corners
                return op.nn.upsampling(data, scale_h, scale_w, data_layout, method, align_corners)
            # elif call.op == self.reshape and len(input_types[0].shape) == 4:
            #     (data,) = new_args
            #     data = op.transpose(data, axes=(0, 4, 1, 5, 2, 3))
            #     new_shape = [int(x) for x in input_types[0].shape]
            #     # Check if the reshape match with such shape after pad
            #     pad, new_shape[1] = _channel_const_match(new_shape[1], self.cfactor)
            #     data = op.reshape(data, new_shape)
            #     # remove pad data
            #     if pad != 0:
            #         new_pad_width = [[0, 0], [0, -pad], [0, 0], [0, 0]]
            #         data = op.nn.pad(data, pad_width=new_pad_width)
            #     return data
            # elif call.op == self.pad:
            #     pad_width = call.attrs.pad_width
            #     if len(pad_width) == 6:
            #         pass
            #     elif len(pad_width) == 4:
            #         (data, pad_value) = args
            #         new_pad_width = []
            #         new_pad_width.extend(pad_width)
            #         for _ in range(2):
            #             new_pad_width.append([0, 0])
            #         return op.nn.pad(data, pad_value=pad_value, pad_width=new_pad_width)
        call = Call(new_fn, new_args, call.attrs, call.type_args, call.span)
        return run_opt_pass(call, relay.transform.InferType())

@tvm.relay.transform.function_pass(opt_level=1)
class WithVTAFunctionTransform:
    """为融合函数 vta_conv2d 添加 ConvAttrs 属性"""
    def __init__(self):
        self.reset()

    def reset(self):
        self._func_index = 0

    def transform_function(self, func, mod, ctx):
        obj = self

        class Replace(ExprMutator):
            def visit_call(self, call):
                new_fn = self.visit(call.op)
                new_args = [self.visit(arg) for arg in call.args]
                attrs = call.attrs
                if isinstance(new_fn, Function):
                    transform = ConvAttrsTransform()
                    transform.visit(new_fn)
                    if transform.attrs: # 添加卷积属性
                        new_fn = new_fn.with_attr("ConvAttrs", transform.attrs[0])
                    func_name = f"{new_fn.attrs['Composite']}__{obj._func_index}"
                    mod[func_name] = new_fn
                    new_fn = mod.get_global_var(func_name)
                    obj._func_index += 1
                call = Call(new_fn, new_args, attrs, call.type_args, call.span)
                return call

        return Replace().visit(func)

def graph_pack(new_fn: Function, bfactor: int, cfactor: int, weight_bits: int):
    """pack new_fn 以支持 VTA"""
    transform = ExprGraphPack(bfactor, cfactor, weight_bits)
    # new_fn = run_mod["main"]
    new_fn = transform.visit(new_fn)
    new_body = new_fn.body
    new_fn = Function(
        list(new_fn.params), new_body,
        ret_type=new_body.checked_type,
        type_params=new_fn.type_params,
        attrs=new_fn.attrs,
        span=new_fn.span
    )
    return run_opt_pass(new_fn, relay.transform.InferType())
