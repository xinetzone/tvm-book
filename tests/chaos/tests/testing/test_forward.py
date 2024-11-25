import os
import platform
import sys
import numpy as np

import torch
from torch.nn import Module
from torch.nn import functional as F
import torchvision

import tvm
import tvm.testing
from tvm import relay
from tvm.contrib import graph_executor
from tvm.contrib.nvcc import have_fp16
from tvm.contrib import cudnn, utils
from .relay.utils.tag_span import _create_span, _set_span, _verify_structural_equal_with_span

sys.setrecursionlimit(10000)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

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

def assert_shapes_match(tru, est):
    """Verfiy whether the shapes are equal"""
    if tru.shape != est.shape:
        msg = "Output shapes {} and {} don't match"
        raise AssertionError(msg.format(tru.shape, est.shape))
    
def load_torchvision(model_name):
    """Given a model name, returns a Torchvision model in eval mode as well
    as an example input."""
    with torch.no_grad():
        if model_name.startswith("inception"):
            height = width = 299
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            height = width = 224
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        input_shape = [1, 3, height, width]
        input_data = torch.randn(input_shape).float()
        for channel in range(3):
            input_data[:, channel] -= mean[channel]
            input_data[:, channel] /= std[channel]

        if model_name.startswith("googlenet"):
            model = getattr(torchvision.models, model_name)(pretrained=True, aux_logits=True)
        else:
            model = getattr(torchvision.models, model_name)(pretrained=True)
        model = model.float().eval()
        return model, [input_data]


def gen_ir_module(model, inputs, use_parser_friendly_name=False):
    """Helper function to generate IRModule with meaningful source information"""

    trace = torch.jit.trace(model, inputs)
    input_names = ["input{}".format(idx) for idx, _ in enumerate(inputs)]
    input_shapes = list(zip(input_names, [inp.shape for inp in inputs]))
    mod, _ = relay.frontend.from_pytorch(
        trace,
        input_shapes,
        use_parser_friendly_name=use_parser_friendly_name,
    )
    return mod
