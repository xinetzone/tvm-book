from tvm import relay
from tvm.relay.dataflow_pattern import (
    wildcard, is_op,
    # FunctionPattern,
    DFPatternCallback,
    # rewrite
)
from .op import special_preprocess

class PreprocessRewrite(DFPatternCallback):
    def __init__(self, a_min=-127.0, a_max=127.0, dtype="int8"):
        super().__init__()
        self.a_min = a_min
        self.a_max = a_max
        self.dtype = dtype
        self.x = wildcard()
        self.scale = wildcard()
        self.multiply = is_op("multiply")(self.x, self.scale)
        self.round = is_op("round")(self.multiply)
        self.clip = is_op("clip")(self.round)
        self.cast = is_op("cast")(self.clip)
        self.pattern = self.cast

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        scale = node_map[self.scale][0]
        return special_preprocess(x, scale, self.a_min, self.a_max, self.dtype)
