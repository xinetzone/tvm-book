from tvm.relay.dataflow_pattern import (
    DFPatternCallback, 
    is_op, wildcard,
    is_constant,
    # is_tuple, 
    # is_tuple_get_item
)
from tvm import relay
from tvm.relay.dataflow_pattern import rewrite
from ..op import special_hard_sigmoid

# ==========================================================================================================================================
def make_hard_sigmoid_v1_pattern():
    x = wildcard()
    r = is_op("add")(x, is_constant())
    r = is_op("clip")(r)
    r = r.has_attr({"a_min": 0., "a_max": 6.0})
    r = is_op("divide")(r, is_constant())
    return r

class HardSigmoidV1Rewrite(DFPatternCallback):
    """融合 mod 中 (clip(x+3,0,6)/6) -> hard_sigmoid(x)"""
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.value_3 = is_constant() # 3
        self.add = is_op("add")(self.x, self.value_3)
        self.clip = is_op("clip")(self.add).has_attr({"a_min": 0., "a_max": 6.0})
        self.value_6 = is_constant() # 6
        self.divide = is_op("divide")(self.clip, self.value_6)
        self.pattern = self.divide

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        value_3 = node_map[self.value_3][0]
        value_6 = node_map[self.value_6][0]
        if (value_3.data.numpy() == 3.0 
            and value_6.data.numpy() == 6.0):
            x = special_hard_sigmoid(x)
            _ = relay.transform.InferTypeLocal(x)
            return x
        else:
            return post

class HardSigmoidV2Rewrite(DFPatternCallback):
    """融合 mod 中 (clip(x*1/6+0.5, 0, 1) -> hard_sigmoid(x)"""
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.value = is_constant() # 1/6
        self.multiply = is_op("multiply")(self.x, self.value)
        self.value2 = is_constant() # 0.5
        self.add = is_op("add")(self.multiply, self.value2)
        self.clip = is_op("clip")(self.add).has_attr({"a_min": 0., "a_max": 1.0})
        self.pattern = self.clip

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        value = node_map[self.value][0]
        value2 = node_map[self.value2][0]
        if ((value.data.numpy()*6 - 1) <= 1e-2
            and value2.data.numpy() == 0.5): 
            relay.transform.InferTypeLocal(x)
            x = special_hard_sigmoid(x)
            _ = relay.transform.InferTypeLocal(x)
            return x
        else:
            return post

def simplify_hard_sigmoid(mod):
    mod["main"] = rewrite(HardSigmoidV1Rewrite(), mod["main"])
    mod["main"] = rewrite(HardSigmoidV2Rewrite(), mod["main"])
    return mod
