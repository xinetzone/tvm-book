
# # PyTorch 量化
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import tvm
from tvm import relay

torch.manual_seed(0)
torch.set_grad_enabled(False)

def list_ops(expr):
    """list_ops"""

    class OpLister(tvm.relay.ExprVisitor):
        """OpLister inherits from ExprVisitor"""

        def visit_op(self, op):
            if op not in self.node_set:
                self.node_list.append(op)
            return super().visit_op(op)

        def list_nodes(self, expr):
            self.node_set = {}
            self.node_list = []
            self.visit(expr)
            return self.node_list

    return OpLister().list_nodes(expr)


class Demo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(16, 64, 3, 1, 1, bias=False, groups=16)
        # self.prelu = nn.PReLU(64)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        # x = self.prelu(x)
        x = self.relu(x)
        return x


class Add1(nn.Module):
    def forward(self, x):
        return x + 1

input_shape = [2]
input_data = torch.rand(input_shape).float()
input_data


compiled_input = {"data": input_data.numpy()}


dev = tvm.cpu()
target = "llvm"
input_shapes = [("data", input_shape)]
model = Add1().float().eval()
trace_model = torch.jit.trace(model, [input_data.clone()])
trace_model = trace_model.float().eval()
mod, params = relay.frontend.from_pytorch(trace_model, input_shapes)
with tvm.transform.PassContext(opt_level=3):
    exe = relay.create_executor(
        "vm", mod=mod, params=params, device=dev, target=target
    ).evaluate()
    result = exe(**compiled_input)


result





