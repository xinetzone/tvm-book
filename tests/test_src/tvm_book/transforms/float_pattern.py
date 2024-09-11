from tvm.relay.dataflow_pattern import (
    wildcard, is_constant, is_op, is_var, is_tuple, is_tuple_get_item
)
from ..special.op import *
# import logging

def is_QPartitionExpr(op):
    r = is_op("annotation.cast_hint")(op)
    r = is_op("annotation.stop_fusion")(r)
    return r

def debug_partition(pattern):
    r = is_op("annotation.cast_hint")(pattern)
    r = is_op("annotation.stop_fusion")(r) | pattern
    return r 

def make_conv_add_squeeze_pattern():
    x = wildcard()
    x = debug_partition(x)
    w = wildcard()
    bias = wildcard()
    x_ = is_op("relay.op.annotation.simulated_quantize")(x, is_constant(), is_constant(), is_constant())
    x_ = x_ | x
    w_ = is_op("relay.op.annotation.simulated_quantize")(w, is_constant(), is_constant(), is_constant())
    w_ = w_ | w
    bias_ = is_op("relay.op.annotation.simulated_quantize")(bias, is_constant(), is_constant(), is_constant()) | bias
    conv_node = is_op("nn.conv2d")(x_, w_)
    r = is_op("add")(conv_node, bias_)
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    r = is_op("squeeze")(r)
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r


def make_conv_add_relu_max_pool2d_pattern():
    x = wildcard()
    x = debug_partition(x)
    w = wildcard()
    bias = wildcard()
    # x_ = is_op("relay.op.annotation.simulated_quantize")(x, is_constant(), is_constant(), is_constant()) | x
    # w_ = is_op("relay.op.annotation.simulated_quantize")(w, is_constant(), is_constant(), is_constant()) | w
    bias_ = is_op("relay.op.annotation.simulated_quantize")(bias, is_constant(), is_constant(), is_constant()) | bias
    conv_node = is_op("nn.conv2d")(x, w)
    r = is_op("add")(conv_node, bias_) | conv_node
    r = is_op("nn.relu")(r) | is_op("nn.prelu")(r, wildcard())
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    xx = is_op("annotation.cast_hint")(r)
    xx = is_op("annotation.stop_fusion")(xx)
    # xx = is_op("nn.max_pool2d")(xx)
    r = is_op("nn.max_pool2d")(xx|r).has_attr({
        "padding": [0, 0, 0, 0],
        # "ceil_mode": False,
    })
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r


def make_conv2d_pattern(x = wildcard()):
    r"""Create a pattern to match the following graph.

    conv2d
        |
    add
        |
        (relu|relu6|prelu|sigmoid|relux)
    """
    x = debug_partition(x)
    w = wildcard()
    bias = wildcard()
    alpha = wildcard()
    # x_ = is_op("relay.op.annotation.simulated_quantize")(x, is_constant(), is_constant(), is_constant()) | x
    # w_ = is_op("relay.op.annotation.simulated_quantize")(w, is_constant(), is_constant(), is_constant()) | w
    bias_ = is_op("relay.op.annotation.simulated_quantize")(bias, is_constant(), is_constant(), is_constant()) | bias
    # alpha_ = is_op("relay.op.annotation.simulated_quantize")(alpha, is_constant(), is_constant(), is_constant()) | alpha
    conv_node = is_op("nn.conv2d")(x, w)
    r = is_op("add")(conv_node, bias_) | is_op("nn.bias_add")(conv_node, bias_) | conv_node
    
    # 激活函数
    r1 = r.optional(lambda x: is_op("nn.relu")(x))
    r2 = r.optional(lambda x: is_op("clip")(x)) # relu6
    r3 = r.optional(lambda x: is_op("nn.prelu")(x, alpha)) # prelu
    r4 = r.optional(lambda x: is_op("sigmoid")(x)) # sigmoid
    r5 = r.optional(lambda x: is_op("multiply")(x, is_op("sigmoid")(x))) # silu
    r6 = r.optional(lambda x: is_op("special.hard_sigmoid")(x)) # special.hard_sigmoid
    r7 = r.optional(lambda x: is_op("special.hard_swish")(x)) # special.hard_swish
    # r6 = r.optional(lambda x: is_op("silu")(x)) # silu
    # r7 = r.optional(lambda x: is_op("hard_sigmoid")(x)) # hard_sigmoid
    # r8 = r.optional(lambda x: is_op("hard_swish")(x)) # hard_swish
    r = r1 | r2 | r3 | r4 | r5 | r6 | r7

    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r


def make_conv2d_transpose_add_activate_pattern():
    r"""Create a pattern to match the following graph.

    conv2d_transpose
        |
    add
    """
    x = wildcard()
    x = debug_partition(x)
    w = wildcard()
    bias = wildcard()
    alpha = wildcard()
    # alpha_ = is_op("relay.op.annotation.simulated_quantize")(alpha, is_constant(), is_constant(), is_constant()) | alpha
    # x_ = is_op("relay.op.annotation.simulated_quantize")(x, is_constant(), is_constant(), is_constant()) | x
    # w_ = is_op("relay.op.annotation.simulated_quantize")(w, is_constant(), is_constant(), is_constant()) | w
    bias_ = is_op("relay.op.annotation.simulated_quantize")(bias, is_constant(), is_constant(), is_constant()) | bias
    r = is_op("nn.conv2d_transpose")(x, w)
    r = is_op("add")(r, bias_) | r
    
    # 激活函数
    r1 = r.optional(lambda x: is_op("nn.relu")(x))
    r2 = r.optional(lambda x: is_op("clip")(x)) # relu6
    r3 = r.optional(lambda x: is_op("nn.prelu")(x, alpha)) # prelu
    r4 = r.optional(lambda x: is_op("sigmoid")(x)) # sigmoid
    r = r1 | r2 | r3 | r4

    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    # r = is_QPartitionExpr(r) | r
    # r = is_op("strided_slice")(r)
    return r


def make_max_pool2d_pattern():
    x = wildcard()
    x = debug_partition(x)
    r = is_op("nn.max_pool2d")(x)
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r


def make_strided_slice_pattern():
    x = wildcard()
    x = debug_partition(x)
    r = is_op("strided_slice")(x)
    return r


def make_concat_pattern():
    x = wildcard()
    x = debug_partition(x)
    r = is_op("concatenate")(x)
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r


def make_avg_pool2d_pattern():
    x = wildcard()
    x = debug_partition(x)
    r = is_op("nn.avg_pool2d")(x) 
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r


def make_adaptive_avg_pool2d_pattern():
    x = wildcard()
    x = debug_partition(x)
    r = is_op("nn.adaptive_avg_pool2d")(x) 
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r


def make_dense_add_pattern():
    r"""Create a pattern to match the following graph.

      nn.dense
        |
       add
    """
    x = wildcard()
    x = debug_partition(x)
    y = wildcard()
    w = wildcard()
    # x_ = is_op("relay.op.annotation.simulated_quantize")(x, is_constant(), is_constant(), is_constant()) | x
    # y_ = is_op("relay.op.annotation.simulated_quantize")(y, is_constant(), is_constant(), is_constant()) | y
    w_ = is_op("relay.op.annotation.simulated_quantize")(w, is_constant(), is_constant(), is_constant()) | w
    node = is_op("nn.dense")(x, y)
    r = is_op("add")(node, w_) | node
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r


def make_add_add_pattern():
    """
    z = x + y
    z2 = z + c1
    其中 c1 为常量
    """
    c1 = is_constant()
    c1_ = is_op("relay.op.annotation.simulated_quantize")(c1, is_constant(), is_constant(), is_constant()) | c1

    r = wildcard() + wildcard()
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    r = r + c1_
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r


def make_multiply_add_pattern():
    """
    z = x * y
    z2 = z + c1
    """
    c1 = is_constant()
    c1_ = is_op("relay.op.annotation.simulated_quantize")(c1, is_constant(), is_constant(), is_constant()) | c1

    r = wildcard() * wildcard()
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    r = r + c1_
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r


def make_add_multiply_add_pattern():
    """
    z = x + y
    z2 = z * c1 + c2
    其中 c1，c2 为常量
    """
    c1 = is_constant()
    c2 = is_constant()
    c1_ = is_op("relay.op.annotation.simulated_quantize")(c1, is_constant(), is_constant(), is_constant()) | c1
    c2_ = is_op("relay.op.annotation.simulated_quantize")(c2, is_constant(), is_constant(), is_constant()) | c2

    r = wildcard() + wildcard()
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    r = r * c1_
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    r = r + c2_
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r


def make_add_pattern():
    r"""Create a pattern to match the following graph.

      add
        |
      relu|relu6
    """
    r = debug_partition(wildcard()) + debug_partition(wildcard())
    r1 = r.optional(lambda x: is_op("nn.relu")(x))
    r2 = r.optional(lambda x: is_op("clip")(x)) # relu6
    r = r1 | r2
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r


def make_multiply_pattern():
    r"""Create a pattern to match the following graph.

      multiply
        |
      relu|relu6
    """
    r = wildcard() * wildcard()
    r1 = r.optional(lambda x: is_op("nn.relu")(x))
    r2 = r.optional(lambda x: is_op("clip")(x)) # relu6
    r = r1 | r2
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r

def make_reshape_squeeze_pattern():
    x = wildcard()
    r = is_op("reshape")(x)
    r = is_op("squeeze")(r)
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r

def make_preprocess_pattern():
    r = is_var()
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r

def make_resize2d_pattern():
    x = wildcard()
    x = debug_partition(x) | x
    return is_op("image.resize2d")(x)

def make_concat_4dim_4tensor_pattern(x0=wildcard(), x1=wildcard(), x2=wildcard(), x3=wildcard(), attrs={}):
    # x0 = wildcard()
    x0_stop_fusion = debug_partition(x0)|x0
    # x1 = wildcard()
    x1_stop_fusion = debug_partition(x1)|x1
    # x2 = wildcard()
    x2_stop_fusion = debug_partition(x2)|x2
    # x3 = wildcard()
    x3_stop_fusion = debug_partition(x3)|x3
    x = is_tuple((x0_stop_fusion, x1_stop_fusion, x2_stop_fusion, x3_stop_fusion))
    x_stop_fusion = debug_partition(x)|x
    if attrs:
        r = is_op("concatenate")(x_stop_fusion).has_attr(attrs)
    else:
        r = is_op("concatenate")(x_stop_fusion)
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r

def make_concat_4dim_3tensor_pattern(x0=wildcard(), x1=wildcard(), x2=wildcard(), attrs={}):
    # x0 = wildcard()
    x0_stop_fusion = debug_partition(x0)|x0
    # x1 = wildcard()
    x1_stop_fusion = debug_partition(x1)|x1
    # x2 = wildcard()
    x2_stop_fusion = debug_partition(x2)|x2
    x = is_tuple((x0_stop_fusion, x1_stop_fusion, x2_stop_fusion))
    x_stop_fusion = debug_partition(x)|x
    if attrs:
        r = is_op("concatenate")(x_stop_fusion).has_attr(attrs)
    else:
        r = is_op("concatenate")(x_stop_fusion)
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r

def make_concat_4dim_2tensor_pattern(x0=wildcard(), x1=wildcard(), attrs={}):
    # x0 = wildcard()
    x0_stop_fusion = debug_partition(x0)|x0
    # x1 = wildcard()
    x1_stop_fusion = debug_partition(x1)|x1
    x = is_tuple((x0_stop_fusion, x1_stop_fusion))
    x_stop_fusion = debug_partition(x)|x
    if attrs:
        r = is_op("concatenate")(x_stop_fusion).has_attr(attrs)
    else:
        r = is_op("concatenate")(x_stop_fusion)
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r

def make_transpose_reshape_concat_softmax_pattern():
    axes = (0, 2, 3, 1)
    x0 = wildcard()
    x1 = wildcard()
    x2 = wildcard()
    transpose0 = is_op("transpose")(x0).has_attr({"axes": axes})
    reshape0 = is_op("reshape")(transpose0) #.has_attr({"newshape": newshape})
    transpose1 = is_op("transpose")(x1).has_attr({"axes": axes})
    reshape1 = is_op("reshape")(transpose1) #.has_attr({"newshape": newshape})
    transpose2 = is_op("transpose")(x2).has_attr({"axes": axes})
    reshape2 = is_op("reshape")(transpose2) #.has_attr({"newshape": newshape})
    tuple_op = is_tuple((reshape0, reshape1, reshape2))
    cat = is_op("concatenate")(tuple_op).has_attr({"axis": 1})
    softmax = is_op("nn.softmax")(cat)
    output = softmax | cat
    return output

def make_transpose_flatten_concat_reshape_softmax_pattern():
    axes = (0, 2, 3, 1)
    x0 = wildcard()
    x1 = wildcard()
    x2 = wildcard()
    transpose0 = is_op("transpose")(x0).has_attr({"axes": axes})
    reshape0 = is_op("reshape")(transpose0).has_attr({"newshape": [0, -1, 1, 1]})
    flatten0 = is_op("squeeze")(reshape0).has_attr({"axis": [2, 3]})
    # flatten0 = is_op("nn.batch_flatten")(transpose0)
    transpose1 = is_op("transpose")(x1).has_attr({"axes": axes})
    reshape1 = is_op("reshape")(transpose1).has_attr({"newshape": [0, -1, 1, 1]})
    flatten1 = is_op("squeeze")(reshape1).has_attr({"axis": [2, 3]})
    # flatten1 = is_op("nn.batch_flatten")(transpose1)
    transpose2 = is_op("transpose")(x2).has_attr({"axes": axes})
    reshape2 = is_op("reshape")(transpose2).has_attr({"newshape": [0, -1, 1, 1]})
    flatten2 = is_op("squeeze")(reshape2).has_attr({"axis": [2, 3]})
    # flatten2 = is_op("nn.batch_flatten")(transpose2)
    tuple_op = is_tuple((flatten0, flatten1, flatten2))
    cat = is_op("concatenate")(tuple_op)
    reshape = is_op("reshape")(cat)
    softmax = is_op("nn.softmax")(reshape)
    output = softmax | reshape
    return output

def make_reshape_pattern(x):
    # x = wildcard()
    x = debug_partition(x) | x
    return is_op("reshape")(x)

def make_split_pattern(x):
    # x = wildcard()
    x = debug_partition(x) | x
    return is_op("split")(x)

def make_softmax_pattern(x, axis=1):
    # x = wildcard()
    x = make_reshape_pattern(x) | x
    r = is_op("relay.op.annotation.simulated_quantize")(x, is_constant(), is_constant(), is_constant())
    softmax = is_op("nn.softmax")(r|x).has_attr({"axis": axis})
    r = is_op("relay.op.annotation.simulated_quantize")(softmax, is_constant(), is_constant(), is_constant())
    return r|softmax

def make_dfl_v1_pattern(x):
    # x = wildcard()
    reshape = make_reshape_pattern(x)
    transpose = is_op("transpose")(reshape).has_attr({"axes": [0, 3, 1, 2]})
    softmax = make_softmax_pattern(transpose, axis=3)
    transpose2 = is_op("transpose")(softmax).has_attr({"axes": [0, 3, 2, 1]})
    conv_weight = is_constant()
    conv_weight = is_op("relay.op.annotation.simulated_quantize")(conv_weight, is_constant(), is_constant(), is_constant()) | conv_weight
    conv = is_op("nn.conv2d")(transpose2, conv_weight).has_attr({
        'strides': [1, 1], 'padding': [0, 0, 0, 0], 'dilation': [1, 1], 
        'groups': 1, 'channels': 1, 'kernel_size': [1, 1], 
        'data_layout': 'NCHW', 'kernel_layout': 'OIHW'
    })
    r = is_op("relay.op.annotation.simulated_quantize")(conv, is_constant(), is_constant(), is_constant())
    r = make_reshape_pattern(conv)
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r

def make_dfl_v2_pattern(x):
    # x = wildcard()
    reshape = make_reshape_pattern(x) | x
    transpose = is_op("transpose")(reshape).has_attr({"axes": [0, 2, 1, 3]})
    softmax = make_softmax_pattern(transpose, axis=1)
    conv_weight = is_constant()
    conv_weight = is_op("relay.op.annotation.simulated_quantize")(conv_weight, is_constant(), is_constant(), is_constant()) | conv_weight
    conv = is_op("nn.conv2d")(debug_partition(softmax)|softmax, conv_weight).has_attr({
        'strides': [1, 1], 'padding': [0, 0, 0, 0], 'dilation': [1, 1], 
        'groups': 1, 'channels': 1, 'kernel_size': [1, 1], 
        'data_layout': 'NCHW', 'kernel_layout': 'OIHW'
    })
    r = is_op("relay.op.annotation.simulated_quantize")(conv, is_constant(), is_constant(), is_constant())
    r = make_reshape_pattern(r|conv)
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r

def make_dfl_v3_pattern(x):
    # x = wildcard()
    reshape = make_reshape_pattern(x)
    softmax = make_softmax_pattern(reshape, axis=1)
    conv_weight = is_constant()
    conv_weight = is_op("relay.op.annotation.simulated_quantize")(conv_weight, is_constant(), is_constant(), is_constant()) | conv_weight
    conv = is_op("nn.conv2d")(softmax, conv_weight).has_attr({
        'strides': [1, 1], 'padding': [0, 0, 0, 0], 'dilation': [1, 1], 
        'groups': 1, 'channels': 1, 'kernel_size': [1, 1], 
        'data_layout': 'NCHW', 'kernel_layout': 'OIHW'
    })
    r = is_op("relay.op.annotation.simulated_quantize")(conv, is_constant(), is_constant(), is_constant())
    r = make_reshape_pattern(conv)
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r

def make_yolo_concat_split_pattern(x11, x12, x21, x22, x31, x32):
    # x11 = wildcard()
    # x12 = wildcard()
    cat1 = make_concat_4dim_2tensor_pattern(x11, x12, {"axis": 1})
    reshape1 = make_reshape_pattern(cat1)

    # x21 = wildcard()
    # x22 = wildcard()
    cat2 = make_concat_4dim_2tensor_pattern(x21, x22, {"axis": 1})
    reshape2 = make_reshape_pattern(cat2)

    # x31 = wildcard()
    # x32 = wildcard()
    cat3 = make_concat_4dim_2tensor_pattern(x31, x32, {"axis": 1})
    reshape3 = make_reshape_pattern(cat3)

    cat = make_concat_4dim_3tensor_pattern(reshape1, reshape2, reshape3, {"axis": 2})
    return make_split_pattern(cat)

def make_yolo_dist2bbox_pattern(x, subtract_anchor_points, add_anchor_points):
    # x = wildcard()
    strided_slice = is_op("strided_slice")(x)
    # anchor_points = is_constant()
    subtract = is_op("subtract")(subtract_anchor_points, strided_slice)
    r1 = is_op("relay.op.annotation.simulated_quantize")(subtract, is_constant(), is_constant(), is_constant()) | subtract
    strided_slice2 = is_op("strided_slice")(x)
    r2 = is_op("relay.op.annotation.simulated_quantize")(add_anchor_points, is_constant(), is_constant(), is_constant()) | add_anchor_points
    add = is_op("add")(strided_slice2, r2) | is_op("add")(r2 | strided_slice2)
    add = is_op("relay.op.annotation.simulated_quantize")(add, is_constant(), is_constant(), is_constant()) | add
    add2 = is_op("add")(r1, add)
    divide_const = is_constant()
    divide = is_op("divide")(add2, divide_const)
    subtract2 = is_op("subtract")(add, subtract)
    tuple_op = is_tuple((divide, subtract2))
    return is_op("concatenate")(tuple_op)

def make_yolo_dist2xywh_pattern(x, anchor_points):
    # x = wildcard()
    # index_max = (1<<31) - 1
    lt = is_op("strided_slice")(x) #.has_attr({"begin": [0, 0], "end": [index_max, 2]})
    rb = is_op("strided_slice")(x) #.has_attr({"begin": [0, 2], "end": [index_max, 4]})
    
    box = is_op("subtract")(rb, lt)
    box_scale = is_op("multiply")(box, is_constant())
    c_xy = is_op("add")(box_scale, anchor_points)
    wh = is_op("add")(rb, lt)
    tuple_op = is_tuple((c_xy, wh))
    return is_op("concatenate")(tuple_op)

def make_yolo_output_all_pattern():
    x0 = wildcard()
    x1 = wildcard()
    x2 = wildcard()
    x3 = wildcard()
    x4 = wildcard()
    x5 = wildcard()
    # x = make_yolo_concat_split_pattern(
    #     make_conv2d_pattern(x0)|x0, 
    #     make_conv2d_pattern(x1)|x1,
    #     make_conv2d_pattern(x2)|x2,
    #     make_conv2d_pattern(x3)|x3,
    #     make_conv2d_pattern(x4)|x4,
    #     make_conv2d_pattern(x5)|x5,
    # )
    x = make_yolo_concat_split_pattern(x0, x1, x2, x3, x4, x5,)

    tuple_get_item_0 = is_tuple_get_item(x, 0)
    yolo_dfl_predict = make_dfl_v1_pattern(tuple_get_item_0) | make_dfl_v2_pattern(tuple_get_item_0) | make_dfl_v3_pattern(tuple_get_item_0)

    anchors = is_constant()
    yolo_dist2bbox_call = make_yolo_dist2xywh_pattern(yolo_dfl_predict, anchors)

    tuple_get_item_1 = is_tuple_get_item(x, 1)
    strides = is_constant()
    multiply = is_op("multiply")(yolo_dist2bbox_call, strides)
    multiply = is_op("annotation.stop_fusion")(multiply) | multiply
    sigmoid = is_op("sigmoid")(tuple_get_item_1)

    tuple_op1 = is_tuple((multiply, sigmoid))
    r = is_op("concatenate")(tuple_op1)
    r = is_op("annotation.stop_fusion")(r) | r
    return r

