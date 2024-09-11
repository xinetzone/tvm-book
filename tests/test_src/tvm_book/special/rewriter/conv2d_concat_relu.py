from tvm import relay
from tvm.relay.dataflow_pattern import (
    wildcard, is_op, is_constant, is_tuple,
    # FunctionPattern,
    DFPatternCallback,
    # rewrite
)

class Conv2dConcatReluRewrite(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.conv1 = is_op("nn.conv2d")(self.x, is_constant())
        self.conv2 = is_op("nn.conv2d")(self.x, is_constant())
        self.conv3 = is_op("nn.conv2d")(self.x, is_constant())
        self.out = is_tuple((self.conv1, self.conv2, self.conv3))
        self.cat = is_op("concatenate")(self.out)
        self.relu = is_op("nn.relu")(self.cat)
        self.pattern = self.relu

    def callback(self, pre, post, node_map):
        # x = node_map[self.x][0]
        conv1 = node_map[self.conv1][0]
        conv1 = relay.nn.relu(conv1)
        conv2 = node_map[self.conv2][0]
        conv2 = relay.nn.relu(conv2)
        conv3 = node_map[self.conv3][0]
        conv3 = relay.nn.relu(conv3)
        out = relay.Tuple((conv1, conv2, conv3))
        cat = node_map[self.cat][0]
        x = relay.concatenate(out, axis=cat.attrs["axis"])
        _ = relay.transform.InferTypeLocal(x)
        return x
