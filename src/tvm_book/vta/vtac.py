import time
from mxnet.gluon.model_zoo import vision

from tvm.ir.transform import PassContext
from tvm import relay
from tvm.contrib import utils
import vta
from vta.top import graph_pack

from .remote import get_remote


class VTAModel:
    def __init__(self, env, device, name="data"):
        self.env = env
        self.device = device
        self.remote = get_remote(self.env)
        self.lib_path = "graphlib.tar"
        self.name = name
        self.dtype_dict = {self.name: "float32"}
        

    @property
    def target(self):
        if self.device == "vta":
            return self.env.target
        else:
            return self.env.target_vta_cpu

    @property
    def ctx(self):
        if self.env.TARGET == "intelfocl":
            return [self.remote.ext_dev(0), self.remote.cpu(0)]
        else:
            # Graph runtime
            if self.device == "vta":
                return self.remote.ext_dev(0)
            else:
                return self.remote.cpu(0)

    def get_model(self, model_name):
        size = (229, 229) if model_name == "inception_v3" else (224, 224)
        shape = 3, *size
        shape_dict = {self.name: (self.env.BATCH, shape)}
        # 开始前端编译
        model = vision.get_model(model_name, pretrained=True)
        return model, shape_dict

    def compile(self, model, shape_dict, start_name, stop_name):
        target = self.target
        # 度量构建的开始时间
        build_start = time.time()
        # 开始前端编
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
        # 更新 shape 和 type 字典
        shape_dict.update({k: v.shape for k, v in params.items()})
        self.dtype_dict.update({k: str(v.dtype) for k, v in params.items()})
        if target.device_name == "vta":
            # 在 Relay 中执行量化
            # 注意：为了 fold batch norm，将 `opt_level` 设置为 `3`
            with PassContext(opt_level=3):
                with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
                    mod = relay.quantize.quantize(mod, params=params)
                # 对 VTA target 进行 graph packing 和 constant folding
                assert self.env.BLOCK_IN == self.env.BLOCK_OUT
                # 如果目标是 intelfocl 或 sim，是否有 device annotation
                relay_prog = graph_pack(
                    mod["main"],
                    self.env.BATCH,
                    self.env.BLOCK_OUT,
                    self.env.WGT_WIDTH,
                    start_name=start_name,
                    stop_name=stop_name,
                    device_annot=(self.env.TARGET == "intelfocl"),
                )
        else:
            relay_prog = mod["main"]

        # 禁用 AlterOpLayout，编译 Relay 程序
        if target.device_name != "vta":
            with PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
                lib = relay.build(
                    relay_prog, target=target, params=params
                )
        else:
            if self.env.TARGET == "intelfocl":
                # 在 CPU 和 VTA 上运行多个目标
                target = {"cpu": self.env.target_vta_cpu,
                            "ext_dev": target}
            with vta.build_config(
                opt_level=3, disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}
            ):
                lib = relay.build(relay_prog, target=target, params=params)

        # 度量 Relay 构建时间
        build_time = time.time() - build_start
        # logging.info(f"{model} inference graph built in {build_time:.2f}s!")

        # 将 inference library 发送到远程 RPC 服务器
        temp = utils.tempdir()
        lib.export_library(temp.relpath(self.lib_path))
        self.remote.upload(temp.relpath(self.lib_path))
        lib = self.remote.load_module(self.lib_path)
        return lib, build_time, shape_dict
