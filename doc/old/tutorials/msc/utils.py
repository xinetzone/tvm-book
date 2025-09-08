import json
import torch
import os
os.environ['PATH'] += ':/usr/local/cuda/bin' # 保证 nvcc 可以被找到

# import tvm.testing
from tvm.contrib.msc.pipeline import MSCManager
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils

def _get_config(
    model_type,
    compile_type,
    tools,
    inputs,
    outputs,
    temp_dir,
    atol=1e-2,
    rtol=1e-2,
    optimize_type=None,
):
    """Get msc config"""

    path = "_".join(["test_tools", model_type, compile_type] + [t["tool_type"] for t in tools])
    return {
        "workspace": msc_utils.msc_dir(temp_dir/path, keep_history=False),
        "verbose": "critical",
        "model_type": model_type,
        "inputs": inputs,
        "outputs": outputs,
        "dataset": {"prepare": {"loader": "from_random", "max_iter": 5}},
        "tools": tools,
        "prepare": {"profile": {"benchmark": {"repeat": 10}}},
        "baseline": {
            "run_type": model_type,
            "profile": {"check": {"atol": atol, "rtol": rtol}, "benchmark": {"repeat": 10}},
        },
        "optimize": {
            "run_type": optimize_type or model_type,
            "profile": {"check": {"atol": atol, "rtol": rtol}, "benchmark": {"repeat": 10}},
        },
        "compile": {
            "run_type": compile_type,
            "profile": {"check": {"atol": atol, "rtol": rtol}, "benchmark": {"repeat": 10}},
        },
    }
def _get_torch_model(name, training=False):
    """Get model from torch vision"""

    # pylint: disable=import-outside-toplevel
    try:
        import torchvision

        model = getattr(torchvision.models, name)()
        if training:
            model = model.train()
        else:
            model = model.eval()
        return model
    except:  # pylint: disable=bare-except
        print("please install torchvision package")
        return None

def _check_manager(manager, expected_info):
    """Check the manager results"""

    model_info = manager.get_runtime().model_info
    passed, err = True, ""
    if not manager.report["success"]:
        passed = False
        err = "Failed to run pipe for {} -> {}".format(manager.model_type, manager.compile_type)
    if not msc_utils.dict_equal(model_info, expected_info):
        passed = False
        err = "Model info {} mismatch with expected {}".format(model_info, expected_info)
    manager.destory()
    if not passed:
        raise Exception("{}\nReport:{}".format(err, json.dumps(manager.report, indent=2)))


def _test_from_torch(
    compile_type,
    tools,
    expected_info,
    temp_dir,
    training=False,
    atol=1e-1,
    rtol=1e-1,
    optimize_type=None,
):
    torch_model = _get_torch_model("resnet50", training)
    if torch_model:
        if torch.cuda.is_available():
            torch_model = torch_model.to(torch.device("cuda:0"))
        config = _get_config(
            MSCFramework.TORCH,
            compile_type,
            tools,
            inputs=[["input_0", [1, 3, 224, 224], "float32"]],
            outputs=["output"],
            temp_dir=temp_dir,
            atol=atol,
            rtol=rtol,
            optimize_type=optimize_type,
        )
        manager = MSCManager(torch_model, config)
        manager.run_pipe()
        _check_manager(manager, expected_info)

def get_model_info(compile_type):
    """Get the model info"""

    if compile_type == MSCFramework.TVM:
        return {
            "inputs": [
                {"name": "input_0", "shape": [1, 3, 224, 224], "dtype": "float32", "layout": "NCHW"}
            ],
            "outputs": [{"name": "output", "shape": [1, 1000], "dtype": "float32", "layout": "NW"}],
            "nodes": {
                "total": 229,
                "input": 1,
                "nn.conv2d": 53,
                "nn.batch_norm": 53,
                "get_item": 53,
                "nn.relu": 49,
                "nn.max_pool2d": 1,
                "add": 16,
                "nn.adaptive_avg_pool2d": 1,
                "reshape": 1,
                "msc.linear_bias": 1,
            },
        }
    if compile_type == MSCFramework.TENSORRT:
        return {
            "inputs": [
                {"name": "input_0", "shape": [1, 3, 224, 224], "dtype": "float32", "layout": "NCHW"}
            ],
            "outputs": [{"name": "output", "shape": [1, 1000], "dtype": "float32", "layout": ""}],
            "nodes": {"total": 2, "input": 1, "msc_tensorrt": 1},
        }
    raise TypeError("Unexpected compile_type " + str(compile_type))
