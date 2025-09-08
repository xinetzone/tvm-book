from tvm.relay.quantize._partition import (
    partition_expr_check,
    QPartitionExpr,
    register_partition_function,
    identity_partition_function,
)
from tvm.relay.quantize.quantize import _forward_op
from tvm.relay.op import op as _op

# def _to_shape(shape):
#     """convert shape into tuple."""
#     return tuple(int(sh) for sh in shape)

_op.get("nn.dense").reset_attr("FQPartitionRewrite")
@register_partition_function("nn.dense")
def dense_partition_function(ref_call, new_args, ctx):
    """Rewrite function for dense for partition"""
    data_cond, data = partition_expr_check(new_args[0])
    kernel_cond, kernel = partition_expr_check(new_args[1])
    assert not kernel_cond
    if data_cond:
        data = new_args[0].realize()
    if data.op.name == "squeeze": # 避免 squeeze+dense 融合
        data = QPartitionExpr(data).realize()
    ret = _forward_op(ref_call, [data, kernel])
    return QPartitionExpr(ret)

# global trace_nodes
# trace_nodes = []
_op.get("nn.max_pool2d").reset_attr("FQPartitionRewrite")
@register_partition_function("nn.max_pool2d")
def max_pool2d_partition_function(ref_call, new_args, ctx):
    padding = ref_call.attrs.padding
    cond, expr = partition_expr_check(new_args[0])
    # print(cond, expr)
    # global trace_nodes
    if cond:
        # trace_nodes.append([padding, ref_call.attrs.ceil_mode, expr])
        # print("pool", set(padding)=={0} and bool(ref_call.attrs.ceil_mode) and expr.op.name == "relu")
        # out_h, out_w = _to_shape(ref_call.checked_type.shape)[2:]
        # data_shape = _to_shape(expr.checked_type.shape)
        # out_h = (data_shape[2] - _to_shape(ref_call.attrs.pool_size)[0]) / _to_shape(ref_call.attrs.strides)[0] + 1
        # out_w = (data_shape[3] - _to_shape(ref_call.attrs.pool_size)[1]) / _to_shape(ref_call.attrs.strides)[1] + 1
        # if (out_h % 1 == 0 and out_w % 1 == 0):
        #     ceil_mode = False
        if set(padding)=={0} and ref_call.attrs.ceil_mode and (expr.op.name == "nn.relu"):
            expr = expr
        else:
            expr = new_args[0].realize() # 保证与 VTA 一致，（resnet18）
        ret = _forward_op(ref_call, [expr])
        # print(f"max_pool2d: {set(padding), not ref_call.attrs.ceil_mode, expr.op.name}\n{ret}")
        return QPartitionExpr(ret)
    return None


def identity_partition_function_two(ref_call, new_args, ctx):
    cond, expr = partition_expr_check(new_args[0])
    if cond:
        expr = new_args[0].realize()
        return QPartitionExpr(_forward_op(ref_call, [expr]))
    return None

_op.get("strided_slice").reset_attr("FQPartitionRewrite")
register_partition_function("strided_slice", identity_partition_function_two)
# _op.get("squeeze").reset_attr("FQPartitionRewrite")
# register_partition_function("squeeze", identity_partition_function_two)
