from tvm.relay.dataflow_pattern import (
    is_constant, is_op, is_tuple, wildcard, 
    is_tuple_get_item
)
# ======================================================================================================
# 辅助函数
# ======================================================================================================
from .common import make_activate

# ======================================================================================================
# 简单算子
# ======================================================================================================
def make_conv2d_bias_pattern():
    """Create a pattern to match the following graph.

      conv2d  bias
        \   /
         add
          |
    (relu|relu6|prelu|sigmoid)
    """
    r = is_op("nn.conv2d")(wildcard(), wildcard())
    r = is_op("add")(r, is_constant())
    # 激活函数
    return make_activate(r)

def make_elwise_pattern():
    """Element-wise 逐元素算子的简称

       + | - | * | /
        |
    (relu|relu6|prelu|sigmoid)
    """
    r1 = wildcard() + wildcard()
    r2 = wildcard() - wildcard()
    r3 = wildcard() * wildcard()
    r4 = wildcard() / wildcard()
    r = r1 | r2 | r3 | r4
    # 激活函数
    return make_activate(r)

def make_dense_bias_pattern():
    r"""Create a pattern to match the following graph.

      x          const
        \       /
        nn.dense
          |
        bias
    """
    node = is_op("nn.dense")(wildcard(), is_constant())
    r = is_op("add")(node, is_constant())
    return r

def make_conv2d_bias_relu_maxpool2d_pattern():
    """Create a pattern to match the following graph.

    conv2d  bias
        \   /
         add
          |
    (relu|relu6|prelu|sigmoid)
          |
        max_pool2d
    """
    r = is_op("nn.conv2d")(wildcard(), is_constant())
    r = is_op("add")(r, is_constant())
    r = is_op("nn.relu")(r)
    r = is_op("nn.max_pool2d")(r).has_attr({
        "padding": [0, 0, 0, 0],
        # "ceil_mode": False,
    })
    return r

def make_pool2d_pattern():
    x = wildcard()
    r1 = is_op("nn.max_pool2d")(x)
    r2 = is_op("nn.adaptive_avg_pool2d")(x)
    return r1 | r2

def make_reshape_pattern():
    x = wildcard()
    r = is_op("reshape")(x)
    return r

def make_squeeze_pattern():
    x = wildcard()
    r = is_op("squeeze")(x)
    return r

# ======================================================================================================
# 复杂算子
# ======================================================================================================
def make_conv2d_bias_add_pattern():
    """Create a pattern to match the following graph.

    conv2d  bias
        \   /
         add
          |
    (relu|relu6|prelu|sigmoid)      other
                              \    /
                               add
    """
    r = is_op("nn.conv2d")(wildcard(), wildcard())
    r = is_op("add")(r, is_constant()) # bias
    r = make_activate(r)
    r = is_op("add")(r, wildcard()) # 残差
    return make_activate(r)