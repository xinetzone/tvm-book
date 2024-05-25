from tvm import relay
from tvm.relay.dataflow_pattern import (
    wildcard, is_op,
    # FunctionPattern,
    DFPatternCallback,
    # rewrite
)
from .op import special_softmax_reshape


class Reshape4dSoftmaxReshape2dRewrite(DFPatternCallback):
    """简化 `reshape4d_softmax_reshape2d` 为 `softmax_reshape`

    原始mod:
    ```
    def @main(%data: Tensor[(1, 3, 8, 8), float32] /* ty=Tensor[(1, 3, 8, 8), float32] span=/conv/Conv.data:0:0 */) -> Tensor[(1, 8), float32] {
        %0 = nn.conv2d(%data, meta[relay.Constant][0] /* ty=Tensor[(8, 3, 1, 1), float32] span=/conv/Conv.conv.weight:0:0 */, padding=[0, 0, 0, 0], channels=8, kernel_size=[1, 1]) /* ty=Tensor[(1, 8, 8, 8), float32] span=/conv/Conv:0:0 */;
        %1 = nn.global_avg_pool2d(%0) /* ty=Tensor[(1, 8, 1, 1), float32] span=/pool/GlobalAveragePool:0:0 */;
        %2 = reshape(%1, newshape=[1, 1, 1, 8]) /* ty=Tensor[(1, 1, 1, 8), float32] span=/Reshape:0:0 */;
        %3 = nn.softmax(%2, axis=3) /* ty=Tensor[(1, 1, 1, 8), float32] span=/Softmax:0:0 */;
        reshape(%3, newshape=[-1, 8]) /* ty=Tensor[(1, 8), float32] span=/Reshape_1:0:0 */
        }
    ```
    简化后为：
    ```
    def @main(%data: Tensor[(1, 3, 8, 8), float32] /* ty=Tensor[(1, 3, 8, 8), float32] span=/conv/Conv.data:0:0 */) -> Tensor[(1, 8), float32] {
        %0 = nn.conv2d(%data, meta[relay.Constant][0] /* ty=Tensor[(8, 3, 1, 1), float32] span=/conv/Conv.conv.weight:0:0 */, padding=[0, 0, 0, 0], channels=8, kernel_size=[1, 1]) /* ty=Tensor[(1, 8, 8, 8), float32] span=/conv/Conv:0:0 */;
        %1 = nn.global_avg_pool2d(%0) /* ty=Tensor[(1, 8, 1, 1), float32] span=/pool/GlobalAveragePool:0:0 */;
        softmax_reshape(%1, __dict__={"axis"=1, "newshape"=[1, 8]}) /* ty=Tensor[(1, 8), float32] */
        }
    ```
    """
    def __init__(self):
        super().__init__()
        self.x = wildcard()
        self.reshape4d = is_op("reshape")(self.x) # 将 NCHW 转换为 NHWC，其他 H=W=1
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
    
