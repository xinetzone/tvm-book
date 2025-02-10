import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    wildcard, is_op, 
    is_constant,
    DFPatternCallback,
    rewrite
)

class L2NormalizeONNX(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.multiply = is_op("multiply")(self.x, self.x)
        self.sum = is_op("sum")(self.multiply)
        self.sqrt = is_op("sqrt")(self.sum)
        self.broadcast_to = is_op("broadcast_to")(self.sqrt)
        self.divide = is_op("divide")(self.x, self.broadcast_to)
        
        self.pattern = self.divide

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        sum = node_map[self.sum][0]
        ret = relay.nn.l2_normalize(x, eps=1e-12, axis=sum.attrs.axis)
        relay.transform.InferTypeLocal(ret)
        return ret

class L2NormalizeTorch(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.abs = is_op("abs")(self.x)
        self.n1  = is_constant()
        self.power = is_op("power")(self.abs, self.n1)
        self.sum = is_op("sum")(self.power)
        self.n2  = is_constant()
        self.sqrt = is_op("power")(self.sum, self.n2)
        self.clip = is_op("clip")(self.sqrt)
        self.broadcast_to_like = is_op("broadcast_to_like")(self.clip, self.x)
        self.divide = is_op("divide")(self.x, self.broadcast_to_like)
        
        self.pattern = self.divide

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        n1 = node_map[self.n1][0]
        n2 = node_map[self.n2][0]
        clip = node_map[self.clip][0]
        dtype = relay.transform.InferTypeLocal(x).dtype
        if (n1.data.numpy() == 2) and (n2.data.numpy()==0.5) and clip.attrs.a_max==np.finfo(dtype).max:
            sum = node_map[self.sum][0]
            ret = relay.nn.l2_normalize(x, eps=clip.attrs.a_min, axis=sum.attrs.axis)
            relay.transform.InferTypeLocal(ret)
            return ret
        return post

@tvm.transform.module_pass(opt_level=1)
class FuseL2Normalize:
    """融合 torch.nn.functional.normalize"""
    def transform_module(self, mod, ctx):
        # 融合 torch.nn.functional.normalize 的 TorchScript 版本
        mod["main"] = rewrite(L2NormalizeTorch(), mod["main"])
        # 融合 torch.nn.functional.normalize 的 ONNX 版本
        mod["main"] = rewrite(L2NormalizeONNX(), mod["main"])
        return mod
