from tvm import relay
from tvm.relay.dataflow_pattern import (
    wildcard, is_op,
    # FunctionPattern,
    DFPatternCallback,
    # rewrite
)
from ..op import special_softmax_reshape


class Reshape4dSoftmaxReshape2dRewrite(DFPatternCallback):
    """简化 `reshape4d_softmax_reshape2d` 为 `softmax_reshape`
    """
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.reshape4d = is_op("reshape")(self.x) # 将 NCHW 转换为 NHWC，其中 H=W=1
        self.softmax = is_op("nn.softmax")(self.reshape4d)
        self.softmax_axis = self.softmax.has_attr({"axis": 3})
        self.reshape2d = is_op("reshape")(self.softmax_axis)
        self.pattern = self.reshape2d

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        b, c, h, w = relay.transform.InferTypeLocal(x).shape
        # assert h == w == 1, ValueError("当前仅仅支持 h == w == 1")
        return special_softmax_reshape(x, axis=1, newshape=(b, c*h*w))

class DefuseSoftmaxReshape(DFPatternCallback):
    """分解 `softmax_reshape`
    """
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.special_softmax_reshape = is_op("special.softmax_reshape")(self.x)
        self.pattern = self.special_softmax_reshape

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        special_softmax_reshape = node_map[self.special_softmax_reshape][0]
        x = relay.nn.softmax(x, axis=int(special_softmax_reshape.attrs.axis))
        x = relay.reshape(x, newshape=special_softmax_reshape.attrs.get_int_tuple("newshape"))
        _ = relay.transform.InferTypeLocal(x)
        return x
    
