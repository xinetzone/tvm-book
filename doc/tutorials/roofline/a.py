import csv
import json
import os
import platform
from io import StringIO

import numpy as np

import tvm.testing
import tvm.utils
from tvm import relay, rpc
from tvm.contrib import utils
from tvm.contrib.debugger import debug_executor
from tvm.relay.testing import mlp
from tvm.runtime import profiler_vm
from tvm.runtime.profiling import Report
from tvm.script import tir as T

for dtype in ["float32", "int8", "int32"]:
    server = rpc.Server(key="roofline_flops_cpu")
    remote = rpc.connect("127.0.0.1", server.port, key="roofline_flops_cpu")
    target = tvm.target.Target("llvm -mattr=+fma,+avx2")
    dev = remote.device(str(target))
    # This test uses vectorized instructions so we need a target that supports them
    flops = tvm.utils.roofline.x86.estimate_peak_fma_vector_flops(target, dev, remote, dtype)
    # Assume we can achieve 1 GFLOP/s per thread, which is 1 FLOP per cycle on a 1GHz cpu.
    assert (
        flops > 10**9 and flops < 10**14
    ), f"FLOP/s should be between 10^9 and 10^14, but it is {flops}"
    print(f"FLOP/s for {dtype} is {flops}")

server = rpc.Server(key="roofline_flops_gpu")
remote = rpc.connect("127.0.0.1", server.port, key="roofline_flops_gpu")
target = tvm.target.Target("cuda")
dev = remote.device(str(target))
# This test uses vectorized instructions so we need a target that supports them
flops = tvm.utils.roofline.cuda.estimate_peak_flops_tensorcore(target, dev, remote)
# should be able to hit a TFLOP/s with tensor cores
assert (
    flops > 10**12 and flops < 10**14
), f"FLOP/s should be between 10^12 and 10^14, but it is {flops}"

# this test should run on all gpus
flops = tvm.utils.roofline.cuda.estimate_peak_flops_fma(target, dev, remote, "float32")
# most gpus since 2016 should be able to hit a TFLOP/s with fma instructions
assert (
    flops > 10**12 and flops < 10**14
), f"FLOP/s should be between 10^12 and 10^14, but it is {flops}"