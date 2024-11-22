from tvm.relay.dataflow_pattern import (
    # TuplePattern, TupleGetItemPattern, 
    is_op, wildcard, is_constant
)
def preprocessing_pattern():
    r = is_op("multiply")(wildcard(), is_constant())
    r = is_op("round")(r)
    r = is_op("clip")(r)
    r = is_op("cast")(r)
    return r

def output_pattern():
    r = is_op("cast")(wildcard())
    r = is_op("multiply")(r, is_constant())
    return r

def pad_reshape_transpose_pattern():
    r = is_op("nn.pad")(wildcard(), wildcard())
    r = is_op("reshape")(r) | r
    r = is_op("transpose")(r)
    r = is_op("broadcast_to")(r) | r
    return r

def conv2d_pattern():
    r"""Create a pattern to match the following graph.

    conv2d
        |
        (add)
        |
        (add)
        |
    (relu|relu6|prelu|sigmoid|relux)
    """
    x = wildcard()
    w = wildcard()
    bias = wildcard()
    bias2 = wildcard()
    alpha = wildcard()

    r = is_op("cast")(x)
    r = is_op("annotation.stop_fusion")(r) | r

    x = r | x

    # w = is_op("cast")(w) | w
    # w = is_op("annotation.stop_fusion")(r) | r

    conv_node = is_op("nn.conv2d")(x, w)
    conv_node = is_op("add")(conv_node, bias2) | conv_node
    
    fixed_point_multiply = is_op("fixed_point_multiply")(conv_node)
    fixed_point_multiply = is_op("cast")(fixed_point_multiply)
    conv_node = fixed_point_multiply | conv_node
    r = is_op("add")(conv_node, bias) | conv_node
    
    # 激活函数
    r1 = r.optional(lambda x: is_op("nn.relu")(x))
    r2 = r.optional(lambda x: is_op("clip")(x)) # relu6
    r3 = r.optional(lambda x: is_op("nn.prelu")(x, alpha)) # prelu
    r4 = r.optional(lambda x: is_op("sigmoid")(x)) # sigmoid
    r = r1 | r2 | r3 | r4

    r_s = is_op("relay.op.annotation.simulated_quantize")(r, is_constant(), is_constant(), is_constant()) | r
    r_s = is_op("annotation.cast_hint")(r_s) | r_s

    r_q = is_op("cast")(r)
    r_q = is_op("fixed_point_multiply")(r_q)
    r_q = is_op("clip")(r_q)
    r_q = is_op("cast")(r_q)
    r_q = is_op("cast")(r_q)
    r = r_s | r_q
    r = is_op("annotation.stop_fusion")(r) | r
    return r
