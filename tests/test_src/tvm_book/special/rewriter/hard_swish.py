from tvm.relay.dataflow_pattern import (
    DFPatternCallback, 
    # rewrite,
    is_op, wildcard,
    is_constant,
    # is_tuple, 
    # is_tuple_get_item
)
from tvm import relay
from ..op import special_hard_swish

# ==========================================================================================================================================
# 融合 mod 中 add+clip+divide+multiply 为 hard_swish
# x*(clip(x+3,0,6)/6) -> hard_swish(x)
def make_hard_swish_pattern():
    x = wildcard()
    r = is_op("add")(x, is_constant())
    r = is_op("clip")(r)
    r = r.has_attr({"a_min": 0., "a_max": 6.0})
    r1 = is_op("divide")(r, is_constant())
    r1 = is_op("multiply")(x, r1)
    return r1

class HardSwishSimplify(DFPatternCallback):
    """融合 x*HardSigmoid(x)
    """
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.hard_sigmoid = is_op("special.hard_sigmoid")(self.x)
        self.swish = is_op("multiply")(self.x, self.hard_sigmoid) | is_op("multiply")(self.hard_sigmoid, self.x)
        self.pattern = self.swish

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        relay.transform.InferTypeLocal(x)
        ret = special_hard_swish(x)
        relay.transform.InferTypeLocal(ret)
        return ret
