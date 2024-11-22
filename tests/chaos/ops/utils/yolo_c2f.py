import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    wildcard, is_op, 
    # is_constant, 
    is_tuple,
    is_tuple_get_item,
    # FunctionPattern,
    DFPatternCallback,
    rewrite
)
from tvm.relay import op as _op
from tvm.relay import transform as _transform
from tvm.relay.frontend.common import infer_value

class FuseSplitConcatRewrite(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.split = is_op("split")(self.x).has_attr({"axis": 1})
        self.tuple_get_item0 = is_tuple_get_item(self.split, 0)
        self.tuple_get_item1 = is_tuple_get_item(self.split, 1)
        self.block =  wildcard()
        self.tuple_ = is_tuple([self.tuple_get_item0, self.tuple_get_item1, self.block])
        self.concat = is_op("concatenate")(self.tuple_)
        self.pattern = self.concat 

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        concat = node_map[self.concat][0]
        block = node_map[self.block][0]
        _transform.InferTypeLocal(x)
        if concat.attrs.axis == 1:
            x = relay.concatenate([x, block], axis=concat.attrs.axis)
            _transform.InferTypeLocal(x)
            return x
        else:
            return post

class SplitItem2StridedSliceRewrite(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.split = is_op("split")(self.x).has_attr({"axis": 1})
        self.tuple_get_item1 = is_tuple_get_item(self.split, 1)
        self.pattern = self.tuple_get_item1

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        split = node_map[self.split][0]
        indices_or_sections = split.attrs['indices_or_sections']
        axis = split.attrs['axis']
        shape = _transform.InferTypeLocal(x).shape
        if len(indices_or_sections) == 1:
            begin = [0] * len(shape)
            begin[axis] = int(indices_or_sections[0])
            ret = relay.strided_slice(x, begin=begin, end=shape)
            _transform.InferTypeLocal(ret)
            return ret
        return post


    
@tvm.transform.module_pass(opt_level=1)
class SimplifyYoloC2F:
    """重写YoloC2F"""
    def transform_module(self, mod, ctx):
        expr = rewrite(FuseSplitConcatRewrite(), mod["main"])
        expr = rewrite(SplitItem2StridedSliceRewrite(), expr)
        mod = tvm.IRModule.from_expr(expr)
        # mod = _transform.AnnotateSpans()(mod)
        # print(mod)
        mod = relay.transform.InferType()(mod)
        return mod