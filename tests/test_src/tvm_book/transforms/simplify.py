# import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    wildcard, is_op, 
    # is_constant, 
    # is_tuple,
    # is_tuple_get_item,
    DFPatternCallback,
    rewrite
)
from tvm.relay import transform as _transform


class FastGlobalAvgPoolSimplify(DFPatternCallback):
    """ 简化 reshape+mean+reshape 为 nn.adaptive_avg_pool2d(%x, output_size=[1]) 

    简化 
        def @main(%x: Tensor[(1, 1024, 8, 6), float32] /* ty=Tensor[(1, 1024, 8, 6), float32] span=aten::size_0.x:0:0 */) -> Tensor[(1, 1024, 1, 1), float32] {
            %0 = reshape(%x, newshape=[1, 1024, -1]) /* ty=Tensor[(1, 1024, 48), float32] span=aten::view_0:0:0 */;
            %1 = mean(%0, axis=[-1]) /* ty=Tensor[(1, 1024), float32] span=aten::mean_0:0:0 */;
            reshape(%1, newshape=[1, 1024, 1, 1]) /* ty=Tensor[(1, 1024, 1, 1), float32] span=aten::view_1:0:0 */
            }
    为
        def @main(%x: Tensor[(1, 1024, 8, 6), float32] /* ty=Tensor[(1, 1024, 8, 6), float32] span=aten::size_0.x:0:0 */) {
            nn.adaptive_avg_pool2d(%x, output_size=[1]) /* ty=Tensor[(1, 1024, 1, 1), float32] */
            }
    """
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.reshape3d = is_op("reshape")(self.x)
        self.mean = is_op("mean")(self.reshape3d)
        self.reshape4d = is_op("reshape")(self.mean)
        self.pattern = self.reshape4d

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        mean = node_map[self.mean][0]
        reshape3d = node_map[self.reshape3d][0]
        reshape4d = node_map[self.reshape4d][0]
        x_shape = _transform.InferTypeLocal(x).shape
        mean_shape = _transform.InferTypeLocal(mean).shape
        reshape3d_shape = _transform.InferTypeLocal(reshape3d).shape
        reshape4d_shape = _transform.InferTypeLocal(reshape4d).shape
        assert len(x_shape) == 4
        assert len(reshape3d_shape) == 3 and reshape3d_shape[:2] == x_shape[:2]
        assert len(mean_shape) == 2 and mean_shape[:] == x_shape[:2]
        assert len(reshape4d_shape) == 4 and reshape4d_shape[:2] == x_shape[:2] and reshape4d_shape[2:] == [1, 1]
        ret = relay.nn.adaptive_avg_pool2d(x, output_size=(1, 1))
        _transform.InferTypeLocal(ret)
        return ret
    