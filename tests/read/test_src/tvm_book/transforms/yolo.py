import numpy as np
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
    """
    .. code-block:: python
        def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
            'Transform distance(ltrb) to box(xywh or xyxy).'
            lt, rb = distance.chunk(2, dim)
            x1y1 = anchor_points - lt
            x2y2 = anchor_points + rb
            if xywh:
                c_xy = (x1y1 + x2y2) / 2
                wh = x2y2 - x1y1
                return torch.cat((c_xy, wh), dim)  # xywh bbox
            return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

    等价于：

   .. code-block:: python
        def dist2bbox2(distance, anchor_points, xywh=True, dim=-1):
            'Transform distance(ltrb) to box(xywh or xyxy).'
            lt, rb = distance.chunk(2, dim)
            if xywh:
                box = rb - lt
                wh = rb + lt
                c_xy = box * 0.5 + anchor_points
                return torch.cat((c_xy, wh), dim)  # xywh bbox
            x1y1 = anchor_points - lt
            x2y2 = anchor_points + rb
            return torch.cat((x1y1, x2y2), dim)  # xyxy bbox
    """
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.lt = is_op("strided_slice")(self.x)#.has_attr({"begin": [np.int64(0)], "end": [np.int64(2)]})
        self.rb = is_op("strided_slice")(self.x)#.has_attr({"begin": [np.int64(2)], "end": [np.int64(4)]})
        self.subtract_anchor_points = wildcard()
        self.subtract = is_op("subtract")(self.subtract_anchor_points, self.lt)
        self.add_anchor_points = wildcard()
        self.add = is_op("add")(self.add_anchor_points, self.rb)
        self.add2 = is_op("add")(self.subtract, self.add)
        self.divide_const = is_constant()
        self.divide = is_op("divide")(self.add2, self.divide_const)
        self.subtract2 = is_op("subtract")(self.add, self.subtract)
        self.tuple_op = is_tuple((self.divide, self.subtract2))
        self.cat = is_op("concatenate")(self.tuple_op)
        self.pattern = self.cat

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        lt = node_map[self.lt][0]
        rb = node_map[self.rb][0]
        
        subtract_anchor_points = node_map[self.subtract_anchor_points][0]
        add_anchor_points = node_map[self.add_anchor_points][0]
        divide_const = node_map[self.divide_const][0]
        assert divide_const.data.numpy() == 2.0
        # assert np.testing.assert_allclose(, rtol=1e-07, atol=1e-5)
        res = set(np.unique(subtract_anchor_points.data.numpy() - add_anchor_points.data.numpy()))
        assert res == {0.0}
        _transform.InferTypeLocal(x)
        _transform.InferTypeLocal(lt)
        _transform.InferTypeLocal(rb)
        box = relay.subtract(rb, lt)
        wh = relay.add(rb, lt)
        c_xy = relay.add(relay.multiply(box, relay.const(0.5)), add_anchor_points)
        z = relay.concatenate((c_xy, wh), axis=1)
        return z
