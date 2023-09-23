import logging
import os
import time
from tvm import rpc, autotvm, runtime
import vta


def get_remote(env,
               device_host="192.168.2.99",
               device_port="9091"):
    if env.TARGET not in ["sim", "tsim", "intelfocl"]:
        # 如果设置环境变量，从 tracker 节点获取 remote。
        # 要设置 tracker，您需要遵循“自动调优卷积网络用于 VTA ”教程。
        tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
        tracker_port = os.environ.get("TVM_TRACKER_PORT", None)
        # 否则，如果你有设备，你想直接从 host 编程，
        # 确保你已经设置了下面的变量为你的板的 IP。
        device_host = os.environ.get("VTA_RPC_HOST", device_host)
        device_port = os.environ.get("VTA_RPC_PORT", device_port)
        # 确保使用 RPC=1 编译 TVM
        assert runtime.enabled("rpc")
        if not tracker_host or not tracker_port:
            remote = rpc.connect(device_host, int(device_port))
        else:
            remote = autotvm.measure.request_remote(
                env.TARGET,
                tracker_host,
                int(tracker_port),
                timeout=10000
            )

        # 重新配置 JIT 运行时和 FPGA。
        # 通过将路径传递给 bitstream 文件而不是 None，
        # 您可以使用自己的自定义 bitstream 编程 FPGA。
        reconfig_start = time.time()
        vta.reconfig_runtime(remote)
        vta.program_fpga(remote, bitstream=None)
        reconfig_time = time.time() - reconfig_start
        print(f"Reconfigured FPGA and RPC runtime in {reconfig_time:.2f}s!")
    # 在仿真模式中，在本地托管 RPC 服务器。
    else:
        remote = rpc.LocalSession()
        if env.TARGET in ["intelfocl"]:
            # program intelfocl aocx
            vta.program_fpga(remote, bitstream="vta.bitstream")
    return remote
