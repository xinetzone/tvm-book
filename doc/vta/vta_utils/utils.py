from tvm import relay
from tvm.relay.testing import run_opt_pass
from tvm.relay import op
from tvm.relay.op import op as _op
from vta.top.graphpack import (
    _channel_const_match,
    _get_tensor_type,
)

# def is_float_call(call: Call):
#     """检查此 call 是否属于浮点数运算。
    
#     通常，浮点数运算的 odtype 是 float；
#     特殊情况是 float->int 转换，它遵循以下 op 序列：
#     multiply(float) -> round(float) -> clip(float) -> cast(int);
#     """
#     args = call.args
#     odtype = _get_tensor_type(call)

#     if odtype == "float32":
#         return True

#     if call.op == _op.get("cast"):
#         idtype = _get_tensor_type(args[0])
#         if idtype == "float32":
#             return True

#     return False

def _pad_channel(data, dshape, cfactor):
    """pad 0 以对齐维度 """
    dshape =  list(dshape)
    pad_width_diff, dshape[1] = _channel_const_match(dshape[1], cfactor)
    if pad_width_diff != 0:
        pad_width = len(dshape) * [[0, 0]]
        pad_width[1] = [0, pad_width_diff]
        data = op.nn.pad(data, pad_width)
        data = run_opt_pass(data, relay.transform.InferType())
    return data

# def _pad_const(data, dshape, ci_factor, co_factor):
#     """pad 0 以对齐维度 """
#     dshape =  list(dshape)
#     pad_batch_diff, dshape[0] = _channel_const_match(dshape[0], co_factor)
#     pad_width_diff, dshape[1] = _channel_const_match(dshape[1], ci_factor)
#     if pad_width_diff != 0 or pad_batch_diff != 0:
#         pad_width = len(dshape) * [[0, 0]]
#         if pad_width_diff != 0:
#             pad_width[1] = [0, pad_width_diff]
#         if pad_batch_diff != 0:
#             pad_width[0] = [0, pad_batch_diff]
#         data = op.nn.pad(data, pad_width)
#         data = run_opt_pass(data, relay.transform.InferType())
#     return data

def _pack_batch_channel(data, dshape, bfactor, cfactor):
    """Pack the data channel dimension."""
    assert int(dshape[0]) % bfactor == 0
    assert int(dshape[1]) % cfactor == 0
    assert len(dshape) == 4 or len(dshape) == 2 # NCHW 或者 NC
    dshape =  list(dshape)
    if len(dshape) == 4:
        data = op.reshape(
            data,
            newshape=(
                int(dshape[0]) // bfactor,
                bfactor,
                int(dshape[1]) // cfactor,
                cfactor,
                int(dshape[2]),
                int(dshape[3]),
            ),
        )
        data = op.transpose(data, axes=(0, 2, 4, 5, 1, 3))
    if len(dshape) == 2:
        data = op.reshape(
            data,
            newshape=(
                int(dshape[0]) // bfactor,
                bfactor,
                int(dshape[1]) // cfactor,
                cfactor,
            ),
        )
        data = op.transpose(data, axes=(0, 2, 1, 3))
    data = run_opt_pass(data, relay.transform.InferType())
    return data

def _unpack_batch_channel(data, old_shape, unpack_transpose=False):
    """Unpack the data channel dimension."""
    assert len(old_shape) == 4 or len(old_shape) == 2 # NCHW 或者 NC
    if unpack_transpose:
        if len(old_shape) == 4:
            data = op.transpose(data, axes=(0, 4, 1, 5, 2, 3))
        elif len(old_shape) == 2:
            data = op.transpose(data, axes=(0, 2, 1, 3))
    data = op.reshape(data, newshape=old_shape)
    return data
