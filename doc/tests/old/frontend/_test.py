import numpy as np
from tvm import relay
import tvm
from tvm.relay.dataflow_pattern import (
    wildcard, is_op, 
    is_constant, 
    # is_tuple,
    # FunctionPattern,
    DFPatternCallback,
    rewrite
)
from tvm.relay import op as _op
from tvm.relay import transform as _transform
from tvm.relay.frontend.common import infer_value

@tvm.register_func
def hard_sigmoid(x):
    print(x)
    print(type(x))
    return _op.clip(_op.multiply(x, _op.const(1/6)) + _op.const(0.5), 0.0, 1.0)

class HardSigmoidV1Rewrite(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.cons1 = is_constant()
        self.add = is_op("add")(self.x, self.cons1)
        self.clip = is_op("clip")(self.add)
        self.cons2 = is_constant()
        self.divide = is_op("divide")(self.clip, self.cons2)
        self.pattern = self.divide

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        clip = node_map[self.clip][0]
        _transform.InferTypeLocal(x)
        cons1 = node_map[self.cons1][0]
        cons2 = node_map[self.cons2][0]
        if np.allclose(cons1.data.numpy(), 3.0) and np.allclose(cons2.data.numpy(), 6.0) and np.allclose(clip.attrs.a_min, 0.0) and np.allclose(clip.attrs.a_max, 6.0):
            out = hard_sigmoid(x)
            _transform.InferTypeLocal(out)
        else:
            out = post
        return out

class HardSigmoidV2Rewrite(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.cons1 = is_constant()
        self.multiply = is_op("multiply")(self.x, self.cons1)
        self.cons2 = is_constant()
        self.add = is_op("add")(self.multiply, self.cons2)
        self.clip = is_op("clip")(self.add)
        self.pattern = self.clip

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        clip = node_map[self.clip][0]
        _transform.InferTypeLocal(x)
        cons1 = node_map[self.cons1][0]
        cons2 = node_map[self.cons2][0]
        if np.allclose(cons1.data.numpy(), 0.2) and np.allclose(cons2.data.numpy(), 0.5) and np.allclose(clip.attrs.a_min, 0.0) and np.allclose(clip.attrs.a_max, 1.0):
            out = tvm.get_global_func("hard_sigmoid")(x)
            # out = _op.clip(x, 0, 1)
            _transform.InferTypeLocal(out)
        else:
            out = post
        return out

@tvm.transform.module_pass(opt_level=1)
class FuseHardSigmoid:
    """融合 HardSigmoid"""
    def transform_module(self, mod, ctx):
        mod["main"] = rewrite(HardSigmoidV1Rewrite(), mod["main"])
        mod["main"] = rewrite(HardSigmoidV2Rewrite(), mod["main"])
        return mod

mod = FuseHardSigmoid()(model.mod)
mod.show()
simplify = relay.transform.SimplifyInference()
mod = relay.transform.InferType()(mod)
mod = simplify(mod)
print(mod)

from tvm.contrib.msc.core.frontend import translate
from tvm.contrib.msc.framework.tvm import codegen as tvm_codegen


opt_config = {"opt_level": 3}
graph, weights = translate.from_relay(model.mod, model.params, opt_config=opt_config)
codegen_config = {"explicit_name": False, "from_relay": True}
rt_mod = tvm_codegen.to_relax(graph, weights, codegen_config)
rt_mod.show()