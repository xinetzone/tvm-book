"""Relay 实现的 graph packing"""

import tvm
from tvm import relay
from tvm.relay import op, transform
from tvm.relay import ExprMutator


def _to_shape(shape):
    """convert shape into tuple."""
    return tuple(int(sh) for sh in shape)


def _pack_batch_channel(data, dshape, bfactor, cfactor):
    """Pack the data channel dimension."""
    assert int(dshape[0]) % bfactor == 0
    assert int(dshape[1]) % cfactor == 0
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
    return data


def _unpack_batch_channel(data, old_shape, unpack_transpose=False):
    """Unpack the data channel dimension."""
    if unpack_transpose:
        data = op.transpose(data, axes=(0, 4, 1, 5, 2, 3))
    data = op.reshape(data, newshape=old_shape)
    return data


def _channel_const_match(channel_length, cfactor_out):
    """Round the channel const variant if the value not divisible by cfactor_out"""
    diff = int(channel_length) % cfactor_out
    if diff != 0:
        diff = cfactor_out - diff
        channel_length = channel_length + diff

    return diff, channel_length


def _const_shape_match(data, dshape, cfactor_out):
    """Pad the constant if the shape[0] not divisible by cfactor_out."""
    assert len(dshape) == 3
    pad_width = int(dshape[0]) % cfactor_out
    if pad_width != 0:
        pad_width = cfactor_out - pad_width
        data = op.nn.pad(data, [[0, pad_width], [0, 0], [0, 0]])
        dshape = tuple([dshape[0] + pad_width, dshape[1], dshape[2]])
    return data, dshape


def _weight_shape_match(data, dshape, channels, cfactor_out, transpose=False):
    """Pad the weight if the shape[0] not divisible by cfactor_out."""
    assert len(dshape) == 4
    pad_width = int(dshape[0]) % cfactor_out
    channels_pad = int(channels) % cfactor_out
    if pad_width != 0:
        pad_width = cfactor_out - pad_width
        data = op.nn.pad(data, [[0, pad_width], [0, 0], [0, 0], [0, 0]])
        dshape = tuple([dshape[0] + pad_width, dshape[1], dshape[2], dshape[3]])

    if channels_pad != 0:
        channels = channels + (cfactor_out - channels_pad)

    return data, dshape, channels


def _weight_shape_match_transpose(data, dshape, channels, cfactor_out):
    """Pad the weight if the shape[1] not divisible by cfactor_out."""
    assert len(dshape) == 4
    pad_width = int(dshape[1]) % cfactor_out
    channels_pad = int(channels) % cfactor_out
    if pad_width != 0:
        pad_width = cfactor_out - pad_width
        data = op.nn.pad(data, [[0, 0], [0, pad_width], [0, 0], [0, 0]])
        dshape = tuple(dshape[0], [dshape[1] + pad_width, dshape[2], dshape[3]])

    if channels_pad != 0:
        channels = channels + (cfactor_out - channels_pad)

    return data, dshape, channels


def _pack_weight(data, dshape, cfactor):
    """Pack the weight into packed format."""
    assert len(dshape) == 4
    assert int(dshape[0]) % cfactor == 0
    assert int(dshape[1]) % cfactor == 0
    data = op.reshape(
        data,
        newshape=(
            int(dshape[0]) // cfactor,
            cfactor,
            int(dshape[1]) // cfactor,
            cfactor,
            int(dshape[2]),
            int(dshape[3]),
        ),
    )
    data = op.transpose(data, axes=(0, 2, 4, 5, 1, 3))
    return data


def _pack_weight_conv2d_transpose(data, dshape, cfactor):
    """Pack the weight into packed format."""
    dshape = _to_shape(dshape)
    assert len(dshape) == 4
    assert dshape[0] % cfactor == 0
    assert dshape[1] % cfactor == 0
    data = op.reshape(
        data,
        newshape=(
            dshape[0] // cfactor,
            cfactor,
            dshape[1] // cfactor,
            cfactor,
            dshape[2],
            dshape[3],
        ),
    )
    data = op.transpose(data, axes=(2, 0, 4, 5, 3, 1))
    return data


def _pack_const(data, dshape, dtype, bfactor, cfactor):
    """Pack a constant parameter."""
    dshape = _to_shape(dshape)
    assert len(dshape) == 3
    assert dshape[0] % cfactor == 0
    data = op.reshape(data, newshape=(dshape[0] // cfactor, cfactor, dshape[1], dshape[2], 1))
    data = op.transpose(data, axes=(0, 2, 3, 4, 1))

    # broadcast batch dimension to bfactor
    data = op.broadcast_to(
        data, shape=(dshape[0] // cfactor, dshape[1], dshape[2], bfactor, cfactor)
    )
    return data


def _get_tensor_shape(node):
    """Get node shape."""
    if isinstance(node.checked_type, relay.ty.TensorType):
        return _to_shape(node.checked_type.shape)
    return []


def _get_tensor_type(node):
    """Get node type."""
    if isinstance(node.checked_type, relay.ty.TensorType):
        return node.checked_type.dtype
    return "float32"
