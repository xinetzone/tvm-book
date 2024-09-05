from tvm.relay.dataflow_pattern import is_constant, is_op, wildcard, is_tuple_get_item
# ======================================================================================================
# 辅助函数
# ======================================================================================================
from .common import make_activate

# ======================================================================================================
# 简单算子
# ======================================================================================================
def qmake_preprocess_pattern():
    """预处理模式

      multiply
        |
       round
        |
       clip
        |
       cast
    """
    r = is_op("multiply")(wildcard(), is_constant())
    r = is_op("round")(r)
    r = is_op("clip")(r)
    r = is_op("cast")(r)
    return r

def qmake_output_pattern():
    """后处理模式

      cast
        |
      multiply
    """
    x = wildcard()
    x = is_op("annotation.stop_fusion")(x) | x
    r = is_op("cast")(x)
    r = is_op("multiply")(r, is_constant())
    return r

def qmake_conv2d_pattern():
    """Create a pattern to match the following graph.

      conv2d
        |
       (add)
        |
    (relu|relu6|prelu|sigmoid)
    """
    x = wildcard()
    w = wildcard()
    xx = is_op("clip")(x)
    xx = is_op("cast")(xx)
    xx = is_op("annotation.stop_fusion")(xx)
    x = xx | x
    conv_node = is_op("nn.conv2d")(x, w)
    r1 = is_op("add")(conv_node, is_constant()) 
    r = r1 | conv_node
    r = make_activate(r)
    r = is_op("fixed_point_multiply_uint8_per_axis")(r, is_constant()) | r
    r = is_op("clip")(r)
    r = is_op("cast")(r)
    r1 = is_op("clip")(r)
    r1 = is_op("cast")(r1)
    return r1 | r

def qmake_max_pool2d_pattern():
    x = wildcard()
    # xx = is_op("clip")(x)
    # xx = is_op("cast")(xx)
    # xx = is_op("annotation.stop_fusion")(xx)
    # x = xx | x
    r = is_op("nn.max_pool2d")(x)
    r = is_op("cast")(r)
    return r
