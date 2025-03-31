from torch.ao.quantization import get_default_qconfig_mapping #QConfigMapping
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from draft.resnet_sigmoid import resnet18
from draft.dataset import Cifar10
import tvm
from tvm.relay.frontend import qnn_torch
from tvm.relay.frontend.pytorch import (
    _run_jit_passes,
    Prelude, PyTorchOpConverter,
    get_all_op_names,
    _get_relay_input_vars,
    _debug_rename,
    convert_params,
    _get_output_name,
    get_attr_chains,
    _getattr_full_name,
    _get_users,
    getattr_attr_name,
    _get_tensor_and_var
)


def calibrate(model, data_loader, num=200):
    model.eval()
    with torch.no_grad():
        for k, (image, _) in tqdm(enumerate(data_loader)):
            if k > num:
                break
            model(image)

shape = 1, 3, 32, 32
model = resnet18()
model.conv1 = nn.Conv2d(model.conv1.in_channels, 
                        model.conv1.out_channels,
                        3, 1, 1, bias=False)
model.fc = nn.Linear(model.fc.in_features, 10)
model_path = "draft/params/resnet18_cifar10_sigmoid.h5"
model.load_state_dict(torch.load(model_path), strict=True)
# script_module = torch.jit.trace(model.eval(), torch.rand(*shape))
dataset = Cifar10(root="draft/data", batch_size=1)
trainset = dataset.train_loader() # 训练集
qconfig_mapping = get_default_qconfig_mapping("qnnpack")
example_input, _ = next(iter(trainset))
model = prepare_fx(model.eval(), qconfig_mapping, (example_input,))
calibrate(model, trainset) # 在样本数据上运行校准
script_model = convert_fx(model)
input_infos = [("data", shape),]
mod, params = from_pytorch(
    script_module, input_infos,
    custom_convert_map=None,
    default_dtype='float32',
    use_parser_friendly_name=False,
    keep_quantized_weight=False
)

from torch.ao.quantization import get_default_qconfig_mapping #QConfigMapping
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from draft.resnet_prelu import resnet18
from draft.dataset import Cifar10

def calibrate(model, data_loader, num=200):
    model.eval()
    with torch.no_grad():
        for k, (image, _) in tqdm(enumerate(data_loader)):
            if k > num:
                break
            model(image)

shape = (1, 3, 32, 32)
model = resnet18()
model.conv1 = nn.Conv2d(model.conv1.in_channels, 
                        model.conv1.out_channels,
                        3, 1, 1, bias=False)
model.fc = nn.Linear(model.fc.in_features, 10)
model_path = "draft/params/resnet18_cifar10_prelu.h5"
model.load_state_dict(torch.load(model_path), strict=True)
# script_module = torch.jit.trace(model.eval(), torch.rand(*shape))
dataset = Cifar10(root="draft/data", batch_size=1)
trainset = dataset.train_loader() # 训练集
qconfig_mapping = get_default_qconfig_mapping("qnnpack")
example_input, _ = next(iter(trainset))
model = prepare_fx(model.eval(), qconfig_mapping, (example_input,))
calibrate(model, trainset) # 在样本数据上运行校准
script_model = convert_fx(model)
input_infos = [("data", shape),]
mod, params = from_pytorch(
    script_module, input_infos,
    custom_convert_map=None,
    default_dtype='float32',
    use_parser_friendly_name=False,
    keep_quantized_weight=False
)

mod = tvm.IRModule()
prelude = Prelude(mod)
enable_lower_all_tuples = True

converter = PyTorchOpConverter(prelude, default_dtype, use_parser_friendly_name)
graph = script_module.graph.copy()
graph_inputs = list(graph.inputs())
_run_jit_passes(graph, enable_lower_all_tuples)
op_names = get_all_op_names(graph)
converter.report_missing_conversion(op_names)
is_module = isinstance(script_module, torch.jit.ScriptModule)
params = script_module.state_dict() if is_module else {}
outputs = _get_relay_input_vars(
    graph, input_infos, prelude, default_dtype=default_dtype, is_module=is_module
)
source_map = _debug_rename(graph, use_parser_friendly_name)
param_vars, tensors, packed_param_map, param_debug_name_map = convert_params(
    graph, params, source_map, use_parser_friendly_name
)
tvm_params = {k: tvm.nd.array(v) for k, v in tensors.items()}
outputs.update(param_vars)
quantized_ops = set(["aten::quantize_per_tensor", "quantized::linear_dynamic"])
if len(quantized_ops.intersection(set(op_names))) > 0:
    weight_quant_params = qnn_torch.get_weight_quant_params(
        script_module, packed_param_map.values()
    )
    qnn_torch.inline_input_quant_params_for_fx(graph, tensors, param_debug_name_map)

# state_dict = params
# getattr_nodes = graph.findAllNodes("prim::GetAttr", recurse=True)
# params = {}
# param_tensors = {}
# packed_param_map = {}
# param_debug_name_map = {}
# vars_by_name = {}
# seen = set()
# attr_name_sep = "_" if use_parser_friendly_name else "."

# for node in getattr_nodes:
#     if _get_output_name(node) in seen:
#         continue

#     for getattrs in get_attr_chains(node):
#         seen.update(map(_get_output_name, getattrs))

#         full_attr = _getattr_full_name(getattrs, attr_name_sep)
#         full_attr_node_name = _get_output_name(getattrs[-1])
#         print(full_attr, full_attr_node_name)
#         # set variable name by concatenating first consumer's name with full attribute
#         # e.g. "aten::batch_norm_5.running_mean"
#         var_name = attr_name_sep.join(
#             [source_map[_get_users(getattrs[-1])[0]], full_attr.split(attr_name_sep)[-1]]
#         )

#         if full_attr.endswith("_packed_params"):  # for quantized models
#             packed_param_map[full_attr_node_name] = full_attr
#         elif full_attr in state_dict:
#             if var_name in vars_by_name:
#                 var = vars_by_name[var_name]
#             else:
#                 torch_tensor = state_dict[full_attr]
#                 tensor, var = _get_tensor_and_var(torch_tensor, var_name)
#                 param_tensors[var_name] = tensor
#                 # for quantized parameters to be correctly located
#                 param_debug_name_map[full_attr_node_name] = var_name
#                 vars_by_name[var_name] = var
#             params[full_attr_node_name] = var

def get_full_attr_name(current):
    current_attr = getattr_attr_name(current)
    inputs = list(current.inputs())
    # logging.debug(f"current_attr: {current_attr}")
    if len(inputs) == 1:
        # logging.debug(f"get_full_attr_name(inputs[0].node()): {inputs[0].node()}")
        if inputs[0].node().kind() == "prim::GetAttr":
            return get_full_attr_name(inputs[0].node()) + "." + current_attr
        elif inputs[0].node().kind() == "prim::Param":
            return current_attr + ".1"
    return current_attr
for node in graph.findAllNodes("prim::GetAttr", recurse=True):
    out_name = node.output().debugName()
    if "_scale" in out_name or "_zero_point" in out_name:
        full_attr = param_debug_name_map[get_full_attr_name(node)]
        assert full_attr in params, f"{full_attr} not found in param dict."
        param_np = params[full_attr].asnumpy()
        new_const_node = graph.create("prim::Constant")
        new_const_node.insertBefore(node)
        break