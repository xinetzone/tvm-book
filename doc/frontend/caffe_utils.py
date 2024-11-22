import os
import logging

from google.protobuf import text_format
import caffe
from caffe import layers as L
from caffe.proto import caffe_pb2 as pb

import tvm
import tvm.testing
from tvm import relay
from tvm.contrib import graph_executor

os.environ["GLOG_minloglevel"] = "2"

logging.basicConfig(level=logging.ERROR)

CURRENT_DIR = os.path.join(os.path.expanduser("~"), ".tvm_test_data", "caffe_test")

#######################################################################
# Generic functions for TVM & Caffe
# ------------------------------------------


def _create_dir(d_path):
    """If the directory is not existed, create it"""
    if not (os.path.exists(d_path) and os.path.isdir(d_path)):
        os.makedirs(d_path)


def _list_to_str(ll):
    """Convert list or tuple to str, separated by underline."""
    if isinstance(ll, (tuple, list)):
        tmp = [str(i) for i in ll]
        res = "_".join(tmp)
    return res


def _gen_filename_str(op_name, data_shape, *args, **kwargs):
    """Combining the filename according to the op_name, shape and other args."""
    file_dir = os.path.join(CURRENT_DIR, op_name)
    _create_dir(file_dir)
    res = op_name + "_"
    shape_str = _list_to_str(list(data_shape))
    res += shape_str
    for arg in args:
        if isinstance(arg, (tuple, list)):
            res += "_" + _list_to_str(arg)
        elif isinstance(arg, (int, float, str)):
            res += "_" + str(arg)
    for _, v in kwargs.items():
        if isinstance(v, (tuple, list)):
            res += "_" + _list_to_str(v)
        elif isinstance(v, (int, float, str)):
            res += "_" + str(v)
    res = res.replace(".", "_")
    res = res.replace("-", "_")
    proto_file = os.path.join(file_dir, res + ".prototxt")
    blob_file = os.path.join(file_dir, res + ".caffemodel")
    solver_file = os.path.join(file_dir, res + "_solver.prototxt")

    return (proto_file, blob_file, solver_file)


def _save_prototxt(n_netspec, f_path):
    """Generate .prototxt file according to caffe.NetSpec"""
    s = n_netspec.to_proto()
    with open(f_path, "w") as f:
        f.write(str(s))


def _save_solver(solver_file, proto_file, blob_file):
    """Define a solver proto, you can change the configs."""
    blob_file_prefix = blob_file.split(".caffemodel")[0]
    s = pb.SolverParameter()
    s.train_net = proto_file
    s.base_lr = 0.01
    s.momentum = 0.9
    s.weight_decay = 0.0005
    s.lr_policy = "inv"
    s.gamma = 0.0001
    s.power = 0.75
    s.display = 1
    s.max_iter = 100000
    s.snapshot = 100000
    s.snapshot_prefix = blob_file_prefix

    with open(solver_file, "w") as f:
        f.write(str(s))


def _save_caffemodel(solver_file, blob_file):
    """Generate .caffemodel file."""
    solver = caffe.SGDSolver(solver_file)
    solver.net.save(blob_file)


def _gen_model_files(n_netspec, proto_file, blob_file, solver_file):
    _save_prototxt(n_netspec, proto_file)
    _save_solver(solver_file, proto_file, blob_file)
    _save_caffemodel(solver_file, blob_file)


def _siso_op(data, func, *args, **kwargs):
    """Create single input and single output Caffe op"""
    n = caffe.NetSpec()
    n.data = L.Input(input_param={"shape": {"dim": list(data.shape)}})
    n.output = func(n.data, *args, **kwargs)
    return n


def _miso_op(data_list, func, *args, **kwargs):
    """Create multi input and single output Caffe op"""
    n = caffe.NetSpec()
    if not isinstance(data_list, (tuple, list)):
        raise TypeError(f"Need tuple or list but get {type(data_list)}")
    input_list = []
    for idx, data in enumerate(data_list):
        n["data" + str(idx)] = L.Input(input_param={"shape": {"dim": list(data.shape)}})
        input_list.append(n["data" + str(idx)])
    n.output = func(*input_list, *args, **kwargs)
    return n


def _simo_op(data, func, *args, **kwargs):
    """Create single input and multi output Caffe op"""
    n = caffe.NetSpec()
    n.data = L.Input(input_param={"shape": {"dim": list(data.shape)}})
    output_list = func(n.data, *args, **kwargs)
    for idx, out in enumerate(output_list):
        n["output" + str(idx)] = out
    return n


def _run_caffe(data, proto_file, blob_file):
    """Run caffe model by Caffe according to .caffemodel and .prototxt"""
    net = caffe.Net(proto_file, blob_file, caffe.TEST)
    if isinstance(data, (list, tuple)):
        for idx, d in enumerate(data):
            net.blobs["data" + str(idx)].data[...] = d
    else:
        net.blobs["data"].data[...] = data
    out = net.forward()

    caffe_output = []
    for i in range(len(out.keys())):
        if "output" + str(i) not in out.keys():
            caffe_output.clear()
            return list(out.values())
        caffe_output.append(out["output" + str(i)])
    return caffe_output


def _run_tvm(data, proto_file, blob_file):
    """Run caffe model by TVM according to .caffemodel and .prototxt"""
    init_net = pb.NetParameter()
    predict_net = pb.NetParameter()

    # load model
    with open(proto_file, "r") as f:
        text_format.Merge(f.read(), predict_net)
    # load blob
    with open(blob_file, "rb") as f:
        init_net.ParseFromString(f.read())

    shape_dict = {}
    dtype_dict = {}
    if isinstance(data, (tuple, list)):
        for idx, d in enumerate(data):
            shape_dict["data" + str(idx)] = d.shape
            dtype_dict["data" + str(idx)] = "float32"
    else:
        shape_dict = {"data": data.shape}
        dtype_dict = {"data": "float32"}

    mod, params = relay.frontend.from_caffe(init_net, predict_net, shape_dict, dtype_dict)

    target = "llvm"

    dev = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    dtype = "float32"
    m = graph_executor.GraphModule(lib["default"](dev))
    if isinstance(data, (tuple, list)):
        for idx, d in enumerate(data):
            m.set_input("data" + str(idx), tvm.nd.array(d.astype(dtype)))
    else:
        m.set_input("data", tvm.nd.array(data.astype(dtype)))
    # execute
    m.run()
    tvm_output = []
    # get outputs
    for i in range(m.get_num_outputs()):
        tvm_output.append(m.get_output(i).numpy())
    return tvm_output


def _compare_caffe_tvm(caffe_out, tvm_out, is_network=False):
    for i, _ in enumerate(caffe_out):
        if is_network:
            caffe_out[i] = caffe_out[i][:1]
        tvm.testing.assert_allclose(caffe_out[i], tvm_out[i], rtol=1e-5, atol=1e-5)


def _test_op(data, func_op, op_name, **kwargs):
    """Single op testing pipline."""
    shape_list = []
    if isinstance(data, (list, tuple)):
        n = _miso_op(data, func_op, **kwargs)
        for d in data:
            shape_list.extend(list(d.shape))
    else:
        output_num = 1
        if "ntop" in kwargs:
            output_num = kwargs["ntop"]
        if output_num == 1:
            n = _siso_op(data, func_op, **kwargs)
        else:
            n = _simo_op(data, func_op, **kwargs)
        shape_list = list(data.shape)

    # obtain the .caffemodel file and .prototxt file
    (proto_file, blob_file, solver_file) = _gen_filename_str(op_name, shape_list, **kwargs)
    _gen_model_files(n, proto_file, blob_file, solver_file)
    # run model in Caffe
    caffe_out = _run_caffe(data, proto_file, blob_file)
    # run model in TVM
    tvm_out = _run_tvm(data, proto_file, blob_file)
    _compare_caffe_tvm(caffe_out, tvm_out)

def _test_network(data, proto_file, blob_file):
    # run model in Caffe
    caffe_out = _run_caffe(data, proto_file, blob_file)
    # run model in TVM
    tvm_out = _run_tvm(data, proto_file, blob_file)
    _compare_caffe_tvm(caffe_out, tvm_out, is_network=True)
