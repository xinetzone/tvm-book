import tvm
from tvm import relay
from tvm.relay.expr import Call, Let, Var
from tvm.relay.function import Function, FunctionWithFields
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay import op
from tvm.relay.op import op as _op
from tvm.relay.testing import run_opt_pass
from tvm.relay.dataflow_pattern import (
    # TuplePattern, TupleGetItemPattern, 
    is_op, wildcard, is_constant
)
from tvm.relay import ExprMutator #, ExprVisitor
# from vta.top.graphpack import ExprPack
from vta.top.graphpack import (
    _to_shape,
    # _pack_batch_channel,
    _unpack_batch_channel,
    # _channel_const_match, # 新增
    _const_shape_match,
    _weight_shape_match,
    # _weight_shape_match_transpose, # 新增
    _pack_weight,
    _pack_weight_conv2d_transpose,
    # _pack_const, # 被修改
    _get_tensor_shape,
    _get_tensor_type,
)

class ExprBitPack(ExprMutator):
    """注解算子 pack 和 unpack"""

    def __init__(self, bfactor, cfactor):
        super().__init__()
        self.bfactor, self.cfactor = bfactor, cfactor
        # Cache Operator the algorithm matches against.
        self.bitpack_start = _op.get("annotation.bitpack_start")
        self.bitpack_end = _op.get("annotation.bitpack_end")
        self.multiply = _op.get("multiply")

    def visit_call(self, call: Call):
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        print(type(new_fn))
        # if isinstance(new_fn, Op):
            # print(new_fn)
            # # 
            # # print(new_fn)
            # if new_fn.attrs["Composite"]=="vta_preprocessing":
            #     # call = Call(new_fn, new_args, call.attrs, call.type_args, call.span)
            #     print()
            #     # print(new_fn.body)
            #     # return _pack_batch_channel(call.args[0], oshape, self.bfactor, self.cfactor)
            #     # 
            #     # call = Call(new_fn, new_args, call.attrs, call.type_args, call.span)
                
            #     # print(call)
            #     # return Call(self.bitpack_start, call.args, call.attrs, call.type_args, call.span)
            # if new_fn.attrs["Composite"]=="vta_output":
            #     new_args[0] = Call(self.bitpack_end, [new_args[0]])
        if new_fn == call.op and new_args == list(call.args):
            return call
        return Call(new_fn, new_args, call.attrs, call.type_args, call.span)
    # def visit_let(self, let):
    #     new_var = self.visit(let.var)
    #     new_val = self.visit(let.value)
    #     new_body = self.visit(let.body)
    #     if isinstance(new_val, Function):
    #         if new_val.attrs["Composite"]=="vta_preprocessing":
    #             print(Call(self.bitpack_start, [new_var]))
    #             # new_body = 
    #         # print(type(new_body))
    #     if new_var == let.var and new_val == let.value and new_body == let.body:
    #         return let
    #     return Let(new_var, new_val, new_body)

class PackConv2dMutator(ExprMutator):
    def __init__(self, bfactor, cfactor, weight_bits):
        super().__init__()
        self.bfactor, self.cfactor, self.weight_bits = bfactor, cfactor, weight_bits
    def visit_call(self, call):
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        
        if isinstance(new_fn, Op):
            if call.op.name == "nn.conv2d":
                input_types = [arg.checked_type for arg in new_args]
                # self.nodes.append(call)
                odtype = _get_tensor_type(call)
                if "float" not in odtype:
                    assert 8 % self.weight_bits == 0
                    w_lanes = 8 // self.weight_bits
                    data_layout = "NCHW%dn%dc" % (self.bfactor, self.cfactor)
                    kernel_layout = "OIHW%do%di" % (self.cfactor, self.cfactor)
                    print(data_layout, kernel_layout)
                    data, weight = new_args
                    data_shape = _to_shape(input_types[0].shape)
                    kernel_shape = _to_shape(input_types[1].shape)
                    channels = call.attrs.channels
                    weight, kernel_shape, channels = _weight_shape_match(
                        weight, kernel_shape, channels, self.cfactor
                    )
                    print(channels)
                    # kernel = _pack_weight(weight, kernel_shape, self.cfactor)
        return Call(new_fn, new_args, call.attrs, call.type_args, call.span)


@tvm.relay.transform.function_pass(opt_level=1)
class WithVTAPadTransform:
    """pad 通道以满足 16 对齐"""
    def __init__(self, bfactor, cfactor, weight_bits):
        self.bfactor, self.cfactor, self.weight_bits = bfactor, cfactor, weight_bits
        self.reset()

    def reset(self):
        self.nodes = []

    def transform_function(self, func, mod, ctx):
        obj = self

        class Replace(ExprMutator):
            def visit_call(self, call):
                new_fn = self.visit(call.op)
                new_args = [self.visit(arg) for arg in call.args]
                if isinstance(new_fn, Op):
                    odtype = _get_tensor_type(call)
                    oshape = _get_tensor_shape(call)
                    if "float" not in odtype: # 仅仅处理量化算子
                        # print(new_fn, odtype, oshape)
                        ...
                elif isinstance(new_fn, Function):
                    oshape = list(_to_shape(new_fn.checked_type.ret_type.shape))
                    # new_body = new_fn.body
                    odtype = new_fn.checked_type.ret_type.dtype
                    if "float" not in odtype: # 仅仅处理量化算子
                        # obj.nodes.append(new_fn)
                        new_body = _pack_batch_channel(new_fn.body, oshape, obj.bfactor, obj.cfactor)
                        new_fn = Function(
                            list(new_fn.params), new_body,
                            ret_type=new_body.checked_type,
                            type_params=new_fn.type_params,
                            attrs=new_fn.attrs,
                            span=new_fn.span
                        )
                else:
                    raise ValueError("暂未支持")
                        
                        # input_types = [call.checked_type for arg in new_args]
                    # _pad_channel(data, dshape, self.bfactor, self.cfactor)
                return Call(new_fn, new_args, call.attrs, call.type_args, call.span)

        return Replace().visit(func)

prepare_transform = tvm.transform.Sequential([
    relay.transform.InferType(),
    relay.transform.MergeComposite(pattern_table), # 算子融合
    # WithVTAFunctionTransform(), # 为融合函数 vta_conv2d 添加 ConvAttrs 属性
    # # VTAGraphPackTransform(bfactor, cfactor, weight_bits),
    # relay.transform.InferType(),

])
run_mod = deepcopy(mod)
run_mod = prepare_transform(run_mod)
run_mod = WithVTAPadTransform(bfactor, cfactor, weight_bits)(run_mod)
# transform = ExprPadMutator(bfactor, cfactor, weight_bits)
# new_fn = transform.visit(run_mod["main"])  


# for op_var in run_mod.get_global_vars():
#     if op_var.name_hint == "main":
#         continue
#     new_fn = run_mod[op_var]
#     oshape = list(_to_shape(new_fn.checked_type.ret_type.shape))
#     new_body = new_fn.body
#     odtype = _get_tensor_type(new_body)
#     if "vta_preprocessing" in op_var.name_hint:
#         assert new_fn.attrs["Composite"] == "vta_preprocessing"
#         assert odtype == "int8"
#         new_body = _pack_batch_channel(new_body, oshape, bfactor, cfactor)
#         new_fn = Function(
#             list(new_fn.params), new_body,
#             ret_type=new_body.checked_type,
#             type_params=new_fn.type_params,
#             attrs=new_fn.attrs,
#             span=new_fn.span
#         )
#         run_mod[op_var] = run_opt_pass(new_fn, relay.transform.InferType())
#         break
@tvm.relay.transform.function_pass(opt_level=1)
class WithVTAPadTransform:
    """pad 通道以满足 16 对齐"""
    def __init__(self, bfactor, cfactor, weight_bits):
        self.bfactor, self.cfactor, self.weight_bits = bfactor, cfactor, weight_bits
        self.reset()

    def reset(self):
        self.nodes = []

    def transform_function(self, func, mod, ctx):
        obj = self

        class Replace(ExprMutator):
            def visit_call(self, call):
                new_fn = self.visit(call.op)
                new_args = [self.visit(arg) for arg in call.args]
                if isinstance(new_fn, GlobalVar):
                    obj.nodes.append(call)
                if new_fn == call.op and new_args == list(call.args):
                    return call
                return Call(new_fn, new_args, call.attrs, call.type_args, call.span)
            # def visit_let(self, let):
            #     new_var = self.visit(let.var)
            #     new_val = self.visit(let.value)
            #     new_body = self.visit(let.body)
            #     if isinstance(new_val, Call):
            #         print(new_var.name_hint, type(new_body))
            #     if new_var == let.var and new_val == let.value and new_body == let.body:
            #         return let
            #     return Let(new_var.name_hint, new_val, new_body)

        return Replace().visit(func)

from vta_utils.pack_tool import ExprBitPack as ExprBitPack2
expr = deepcopy(mod["main"])
assert isinstance(expr, relay.Function)
# 融合函数
expr = run_opt_pass(expr, relay.transform.InferType())
transform = ExprBitPack2()
expr = transform.visit(expr)
tvm.IRModule.from_expr(expr).show()