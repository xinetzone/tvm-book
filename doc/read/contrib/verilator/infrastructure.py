"""Verilator 实用函数"""
import os
import sys
import subprocess as sp
import json
from collections.abc import Callable
import tvm
from tvm import relay
import tvm.relay.testing
from tvm import runtime
from tvm.relay import transform

def _register_verilator_op(op_name: str, supported: bool=True) -> Callable:
    """辅助函数，用于指示 Verilator 可以支持给定的算子。

    Args:
        op_name: 将要注册的算子的名称。
        supported: 指示算子是否被 Verilator 支持。
        
    Returns:
        一个装饰器，用于注册算子的支持情况。如果算子受 DNNL 支持，则返回  true 的函数。
    """

    @tvm.ir.register_op_attr(op_name, "target.verilator")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper

def clear_stats():
    """清空 profiler 统计信息"""
    f = tvm.get_global_func("verilator.profiler_clear", True)
    if f:
        f()

def stats():
    """获取 profiler 统计信息"""

    x = tvm.get_global_func("verilator.profiler_status")()
    return json.loads(x)

def offload(mod):
    """基于已注册的 ops 来卸载(Offload) ops

    Paramters
    ---------
    mod : Module
        The input module.

    Returns
    -------
    mod : Module
        The output module with offloaded ops.
    """

    backend = "verilator"
    mod = transform.AnnotateTarget([backend])(mod)
    mod = transform.PartitionGraph()(mod)
    return mod

_register_verilator_op("add")
_register_verilator_op("nn.bias_add")
# tvm.get_global_func("relay.ext.verilator", True)

def compile_hardware(lanes, verilator_app_path):
    """Compile hardware into shared library

    Paramters
    ---------
    lanes : Int
        The number of vector lanes.

    Returns
    -------
    path : Str
        The path of the shared library.
    """
    lib_name = f"libverilator_{lanes}"
    lib_name_ext = f"{lib_name}.so"
    lib = f"{verilator_app_path}/{lib_name_ext}"
    if not os.path.isfile(lib):
        opt_lib_name = f"LIB_NAME={lib_name}"
        opt_lanes = f"LANES={lanes}"
        cmd = []
        cmd.append("make")
        cmd.append("--directory")
        cmd.append(verilator_app_path)
        cmd.append(opt_lib_name)
        cmd.append(opt_lanes)
        sp.run(cmd, check=True, stdout=sp.DEVNULL)
    return lib


def compiler_opts(lib):
    """Create compiler options

    Paramters
    ---------
    lib : Str
        The path of the hardware shared library.

    Returns
    -------
    opts : Dict
        The compiler options.
    """
    opts = {
        "lib_path": lib,
        "profiler_enable": True,
        "profiler_cycle_counter_id": 0,
    }
    return opts


def run_module(inp, mod, params=None, opts=None):
    """Compile Relay module and hardware library

    Paramters
    ---------
    inp : Data
        The input data.

    mod : Module
        The relay module.

    params : Parameters
        The model Parameters.

    opts : Dict
        The compiler

    Returns
    -------
    out : Data
        The output data.
    """

    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.verilator.options": opts}):
        lib = relay.vm.compile(mod, target="llvm", params=params)
    code, lib = lib.save()
    exe = runtime.vm.Executable.load_exec(code, lib)
    vm = runtime.vm.VirtualMachine(exe, tvm.cpu())
    out = vm.run(**inp)
    return out
