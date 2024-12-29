"""pack 卷积算子"""
# import tvm
from tvm.relay import ExprMutator
from tvm.relay.expr import Call
from tvm.ir.op import Op
# from tvm.relay.function import Function
from vta.top.graphpack import (
    _channel_const_match,
    _to_shape,
    _get_tensor_type,
    _pack_weight,
    _weight_shape_match,
)
from tvm.relay.testing import run_opt_pass
from .utils import _pack_batch_channel

class ConvAttrsTransform(ExprMutator):
    """获取融合算子的卷积属性"""
    def __init__(self):
        super().__init__()
        self.attrs = []
    def visit_call(self, call):
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        call = Call(new_fn, new_args, call.attrs, call.type_args, call.span)
        if isinstance(new_fn, Op):
            if call.op.name == "nn.conv2d":
                self.attrs.append(call.attrs)
        return call
