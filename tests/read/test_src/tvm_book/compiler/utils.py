import tvm
from tvm import relay
# from tvm.ir.op import Op
from tvm.relay import Call
from tvm.relay.function import Function
from tvm.relay import transform as _transform
from tvm_book.transforms.float_pattern import *

@tvm.relay.transform.function_pass(opt_level=1)
class AnnotateCompiler:
    """替换融合函数为全局函数并添加编译器属性"""
    def __init__(self, Compiler):
        super().__init__()
        self.Compiler = Compiler
        self.reset()
        
    def reset(self):
        self.func_id = 0
        self.func_names = []
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
                    new_fn = new_fn.with_attr("Compiler", obj.Compiler)
                    mod[func_name] = new_fn
                    obj.func_names.append(func_name)
                    obj.func_id += 1
                    new_fn = mod.get_global_var(func_name)
                call = Call(new_fn, new_args, call.attrs, call.type_args, call.span)
                return call
        return Replace().visit(func)
    

def merge_compiler(mod, compiler_name: str="vta_special"):
    pattern_table = [
        (f"{compiler_name}.yolo_output_all", make_yolo_output_all_pattern()),
        # (f"{compiler_name}.yolo_dfl_predict", make_dfl_v1_pattern()),
        # (f"{compiler_name}.yolo_dfl_predict", make_dfl_v2_pattern()),
        # (f"{compiler_name}.yolo_dfl_predict", make_dfl_v3_pattern()),
        # (f"{compiler_name}.yolo_concat_split", make_yolo_concat_split_pattern(wildcard(), wildcard(), wildcard(), wildcard(), wildcard(), wildcard())),
        # (f"{compiler_name}.yolo_dist2bbox", make_yolo_dist2bbox_pattern(wildcard(), wildcard(), wildcard())),
        # (f"{compiler_name}.preprocess", make_preprocess_pattern()),
        (f"{compiler_name}.transpose_reshape_concat_softmax", make_transpose_reshape_concat_softmax_pattern()),
        (f"{compiler_name}.transpose_flatten_concat_reshape_softmax", make_transpose_flatten_concat_reshape_softmax_pattern()),
        (f"{compiler_name}.concat_4dim_4tensor", make_concat_4dim_4tensor_pattern()),
        (f"{compiler_name}.concat_4dim_3tensor", make_concat_4dim_3tensor_pattern()),
        (f"{compiler_name}.concat_4dim_2tensor", make_concat_4dim_2tensor_pattern()),
        (f"{compiler_name}.conv_add_squeeze", make_conv_add_squeeze_pattern()), # mobilenet_v2_tf 最后一层
        (f"{compiler_name}.conv_add_relu_max_pool2d", make_conv_add_relu_max_pool2d_pattern()),
        (f"{compiler_name}.conv2d_transpose_add_activate", make_conv2d_transpose_add_activate_pattern()),
        (f"{compiler_name}.conv2d", make_conv2d_pattern()),
        (f"{compiler_name}.max_pool2d", make_max_pool2d_pattern()),
        (f"{compiler_name}.dense_add", make_dense_add_pattern()),
        (f"{compiler_name}.adaptive_avg_pool2d", make_adaptive_avg_pool2d_pattern()),
        (f"{compiler_name}.avg_pool2d", make_avg_pool2d_pattern()),
        (f"{compiler_name}.add_multiply_add", make_add_multiply_add_pattern()), # kr_karen
        (f"{compiler_name}.add_add", make_add_add_pattern()),
        (f"{compiler_name}.multiply_add", make_multiply_add_pattern()),
        (f"{compiler_name}.add", make_add_pattern()),
        (f"{compiler_name}.multiply", make_multiply_pattern()),
        (f"{compiler_name}.resize2d", make_resize2d_pattern()),
        # (f"{compiler_name}.vta_softmax", make_softmax_pattern()),
        # (f"{compiler_name}.strided_slice", make_strided_slice_pattern()),
    ]
    merge_passes = tvm.transform.Sequential([
        _transform.InferType(),
        _transform.MergeComposite(pattern_table),
        AnnotateCompiler(compiler_name),
        # _transform.AnnotateTarget(compiler_name),
        # _transform.MergeCompilerRegions(),
        # _transform.PartitionGraph(mod_name='default', bind_constants=True),
        _transform.InferType(),
    ])
    with tvm.transform.PassContext(opt_level=3):
        mod = merge_passes(mod)
    return mod
