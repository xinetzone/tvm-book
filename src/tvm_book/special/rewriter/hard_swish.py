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

class HardSwishRewrite(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.value_3 = is_constant() # 3
        self.add = is_op("add")(self.x, self.value_3)
        self.clip = is_op("clip")(self.add).has_attr({"a_min": 0., "a_max": 6.0})
        self.value_6 = is_constant() # 6
        self.divide = is_op("divide")(self.clip, self.value_6)
        self.multiply = is_op("multiply")(self.x, self.divide)
        self.pattern = self.multiply

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        value_3 = node_map[self.value_3][0]
        assert value_3.data.numpy() == 3.0
        value_6 = node_map[self.value_6][0]
        assert value_6.data.numpy() == 6.0
        x = special_hard_swish(x)
        _ = relay.transform.InferTypeLocal(x)
        return x
