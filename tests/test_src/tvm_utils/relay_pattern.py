from tvm.relay.dataflow_pattern import is_constant, is_op, wildcard
import logging

def is_QPartitionExpr(op):
    r = is_op("annotation.cast_hint")(op)
    r = is_op("annotation.stop_fusion")(r)
    return r

def debug_partition(func):
    def is_QPartitionExpr():
        # logging.debug(f"enter {func.__name__}()")
        r = func()
        r = is_op("annotation.cast_hint")(r) | r
        # r = is_op("annotation.stop_fusion")(r) | r
        return r
    return is_QPartitionExpr

@debug_partition
def make_conv_add_squeeze_pattern():
    x = wildcard()
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

@debug_partition
def make_conv_add_relu_max_pool2d_pattern():
    x = wildcard()
    w = wildcard()
    bias = wildcard()
    # x_ = is_op("relay.op.annotation.simulated_quantize")(x, is_constant(), is_constant(), is_constant()) | x
    # w_ = is_op("relay.op.annotation.simulated_quantize")(w, is_constant(), is_constant(), is_constant()) | w
    bias_ = is_op("relay.op.annotation.simulated_quantize")(bias, is_constant(), is_constant(), is_constant()) | bias
    conv_node = is_op("nn.conv2d")(x, w)
    r = is_op("add")(conv_node, bias_) | conv_node
    r = is_op("nn.relu")(r)
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    r = is_op("nn.max_pool2d")(r).has_attr({
        "padding": [0, 0, 0, 0],
        "ceil_mode": True,
    }) 
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r

@debug_partition
def make_conv_add_activate_pattern():
    r"""Create a pattern to match the following graph.

    conv2d
        |
    (add)
        |
        (relu|relu6|prelu|sigmoid)
    """
    x = wildcard()
    w = wildcard()
    bias = wildcard()
    alpha = wildcard()
    # x_ = is_op("relay.op.annotation.simulated_quantize")(x, is_constant(), is_constant(), is_constant()) | x
    # w_ = is_op("relay.op.annotation.simulated_quantize")(w, is_constant(), is_constant(), is_constant()) | w
    bias_ = is_op("relay.op.annotation.simulated_quantize")(bias, is_constant(), is_constant(), is_constant()) | bias
    alpha_ = is_op("relay.op.annotation.simulated_quantize")(alpha, is_constant(), is_constant(), is_constant()) | alpha
    conv_node = is_op("nn.conv2d")(x, w)
    r = is_op("add")(conv_node, bias_) | conv_node
    
    # 激活函数
    r1 = r.optional(lambda x: is_op("nn.relu")(x))
    r2 = r.optional(lambda x: is_op("clip")(x)) # relu6
    r3 = r.optional(lambda x: is_op("nn.prelu")(x, alpha)) # prelu
    r4 = r.optional(lambda x: is_op("sigmoid")(x)) # sigmoid
    r = r1 | r2 | r3 | r4

    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r

@debug_partition
def make_conv2d_transpose_add_activate_pattern():
    r"""Create a pattern to match the following graph.

    conv2d_transpose
        |
    add
    """
    x = wildcard()
    w = wildcard()
    bias = wildcard()
    alpha = wildcard()
    alpha_ = is_op("relay.op.annotation.simulated_quantize")(alpha, is_constant(), is_constant(), is_constant()) | alpha
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

@debug_partition
def make_max_pool2d_pattern():
    x = wildcard()
    r = is_op("nn.max_pool2d")(x)
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r

@debug_partition
def make_avg_pool2d_pattern():
    x = wildcard()
    r = is_op("nn.avg_pool2d")(x) 
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r

@debug_partition
def make_adaptive_avg_pool2d_pattern():
    x = wildcard()
    r = is_op("nn.adaptive_avg_pool2d")(x) 
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r

@debug_partition
def make_dense_add_pattern():
    r"""Create a pattern to match the following graph.

      nn.dense
        |
       add
    """
    x = wildcard()
    y = wildcard()
    w = wildcard()
    # x_ = is_op("relay.op.annotation.simulated_quantize")(x, is_constant(), is_constant(), is_constant()) | x
    # y_ = is_op("relay.op.annotation.simulated_quantize")(y, is_constant(), is_constant(), is_constant()) | y
    w_ = is_op("relay.op.annotation.simulated_quantize")(w, is_constant(), is_constant(), is_constant()) | w
    node = is_op("nn.dense")(x, y)
    r = is_op("add")(node, w_) | node
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r

@debug_partition
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


@debug_partition
def make_add_pattern():
    r"""Create a pattern to match the following graph.

      add
        |
      relu|relu6
    """
    r = wildcard() + wildcard()
    r1 = r.optional(lambda x: is_op("nn.relu")(x))
    r2 = r.optional(lambda x: is_op("clip")(x)) # relu6
    r = r1 | r2
    r = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    return r

@debug_partition
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
