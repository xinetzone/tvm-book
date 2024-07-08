from tvm import relay
from tvm.relay.dataflow_pattern import (
    wildcard, is_op, 
    is_constant, 
    is_tuple,
    # is_tuple_get_item,
    DFPatternCallback,
    # rewrite
)
from tvm.relay import transform as _transform

class Dist2xywhSimplify(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.strided_slice = is_op("strided_slice")(self.x) #.has_attr({"begin": [0, np.int64(0)]})
        self.strided_slice2 = is_op("strided_slice")(self.x) #.has_attr({"begin": [0, np.int64(2)]})
        self.subtract_anchor_points = wildcard()
        self.subtract = is_op("subtract")(self.subtract_anchor_points, self.strided_slice)
        self.add_anchor_points = wildcard()
        self.add = is_op("add")(self.add_anchor_points, self.strided_slice2)
        self.add2 = is_op("add")(self.subtract, self.add)
        self.divide_const = is_constant()
        self.divide = is_op("divide")(self.add2, self.divide_const)
        self.subtract2 = is_op("subtract")(self.add, self.subtract)
        self.tuple_op = is_tuple((self.divide, self.subtract2))
        self.cat = is_op("concatenate")(self.tuple_op)
        self.pattern = self.cat

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        strided_slice = node_map[self.strided_slice][0]
        strided_slice2 = node_map[self.strided_slice2][0]
        
        subtract_anchor_points = node_map[self.subtract_anchor_points][0]
        add_anchor_points = node_map[self.add_anchor_points][0]
        divide_const = node_map[self.divide_const][0]
        assert divide_const.data.numpy() == 2.0
        # assert np.testing.assert_allclose(, rtol=1e-07, atol=1e-5)
        res = set(np.unique(subtract_anchor_points.data.numpy() - add_anchor_points.data.numpy()))
        assert res == {0.0}
        _transform.InferTypeLocal(x)
        lt = relay.strided_slice(x, **dict(strided_slice.attrs))
        rb = relay.strided_slice(x, **dict(strided_slice2.attrs))
        wh = relay.subtract(rb, lt)
        c_xy = relay.add(relay.multiply(wh, relay.const(0.5)), subtract_anchor_points)
        z = relay.concatenate((c_xy, wh), axis=1)
        return z
