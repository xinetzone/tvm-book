# [`onnx-simplifier`](https://github.com/daquexian/onnx-simplifier) 简化模型：
import onnx
from onnxsim import simplify
onnx_model = onnx.load(f"{output_name}.onnx")  # load onnx model
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, f"{output_name}_s.onnx.onnx")

import onnxoptimizer

optimizers_list = [
    'eliminate_deadend',
    'eliminate_nop_dropout',
    'eliminate_nop_cast',
    'eliminate_nop_monotone_argmax', 'eliminate_nop_pad',
    'extract_constant_to_initializer', 'eliminate_unused_initializer',
    'eliminate_nop_transpose',
    'eliminate_nop_flatten', 'eliminate_identity',
    'fuse_add_bias_into_conv',
    'fuse_consecutive_concats',
    'fuse_consecutive_log_softmax',
    'fuse_consecutive_reduce_unsqueeze', 'fuse_consecutive_squeezes',
    'fuse_consecutive_transposes', 'fuse_matmul_add_bias_into_gemm',
    'fuse_pad_into_conv', 'fuse_transpose_into_gemm', 'eliminate_duplicate_initializer'
    # 'fuse_bn_into_conv',
]

model = onnxoptimizer.optimize(model_simp, optimizers_list, fixed_point=True)
onnx.save(model_simp, f"{output_name}_opt.onnx.onnx")