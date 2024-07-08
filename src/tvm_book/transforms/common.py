import tvm
from tvm import relay
from tvm.ir.expr import GlobalVar
from tvm.relay.op import op as _op
from tvm.relay import Call
from tvm.relay.function import Function



@tvm.relay.transform.function_pass(opt_level=1)
class FuseTransform:
    """替换融合函数为全局函数
    
    .. code-block:: python
        def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
            """Transform distance(ltrb) to box(xywh or xyxy)."""
            lt, rb = distance.chunk(2, dim)
            x1y1 = anchor_points - lt
            x2y2 = anchor_points + rb
            if xywh:
                c_xy = (x1y1 + x2y2) / 2
                wh = x2y2 - x1y1
                return torch.cat((c_xy, wh), dim)  # xywh bbox
            return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

    ``dist2bbox(distance, anchor_points, xywh=True, dim=-1)`` 等价于：

   .. code-block:: python
        def dist2bbox2(distance, anchor_points, xywh=True, dim=-1):
            """Transform distance(ltrb) to box(xywh or xyxy)."""
            lt, rb = distance.chunk(2, dim)
            if xywh:
                wh = rb - lt
                c_xy = wh * 0.5 + anchor_points
                return torch.cat((c_xy, wh), dim)  # xywh bbox
            x1y1 = anchor_points - lt
            x2y2 = anchor_points + rb
            return torch.cat((x1y1, x2y2), dim)  # xyxy bbox
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.func_id = 0
        self.func_names = [] # 记录函数名称
        self.op_names = {} # 统计 op 出现次数

    def transform_function(self, func, mod, ctx):
        obj = self
        class Replace(tvm.relay.ExprMutator):
            def visit_call(self, call):
                new_fn = self.visit(call.op)
                new_args = [self.visit(arg) for arg in call.args]
                if isinstance(call.op, Function):
                    op_name = new_fn.attrs["Composite"]
                    op_id = obj.op_names.get(op_name, 0)
                    new_fn = new_fn.with_attr("op_id", op_id)
                    obj.op_names[op_name] = obj.op_names.get(op_name, 0) + 1
                    func_name = f"{op_name}_{obj.func_id}"
                    new_fn = new_fn.with_attr("func_id", obj.func_id)
                    mod[func_name] = new_fn
                    obj.func_names.append(func_name)
                    obj.func_id += 1
                    new_fn = mod.get_global_var(func_name)
                call = Call(new_fn, new_args, call.attrs, call.type_args, call.span)
                return call
        return Replace().visit(func)

@tvm.relay.transform.function_pass(opt_level=1)
class OutputTransform:
    """将中间全局函数节点转换为输出节点"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.nodes = []

    def transform_function(self, func, mod, ctx):
        obj = self
        class Replace(tvm.relay.ExprMutator):
            def visit_call(self, call):
                new_fn = self.visit(call.op)
                new_args = [self.visit(arg) for arg in call.args]
                call = Call(new_fn, new_args, call.attrs, call.type_args, call.span)
                if isinstance(call.op, GlobalVar):
                    obj.nodes.append(call)
                # print(call)
                return call

        return Replace().visit(func)
