import numpy as np
import torch
from torch import fx
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_fx

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 1, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, data):
        x = self.conv(data)
        return self.relu(x)

def get_model(input_info):
    # 转换前端模型为 IRModule
    with torch.no_grad():
        torch_fx_model = fx.symbolic_trace(M())
        mod = from_fx(torch_fx_model, input_info, keep_params_as_input=False)
    # mod, params = relax.frontend.detach_params(mod)
    return mod, torch_fx_model
