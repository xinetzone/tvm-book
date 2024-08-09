from dataclasses import dataclass
from tvm.driver.tvmc.frontends import lazy_import, load_model
from tvm.driver.tvmc.model import TVMCModel
from tvm import relay

@dataclass
class InferInputConfig:
    name: str
    shape: tuple[int]
    dtype: str
    layout: str # 推理布局 yuv_nv12, yuv_nv21, nhwc, nchw
    mean: int|tuple[int]
    std: int|tuple[int]

@dataclass
class TrainInputConfig:
    name: str
    shape: tuple[int]
    dtype: str
    layout: str # 推理布局 yuv_nv12, yuv_nv21, nhwc, nchw
    
@dataclass
class Config:
    model_type: str
    infer_inputs: list[InferInputConfig]
    train_inputs: list[TrainInputConfig]

    def __post_init__(self):
        self.infer_inputs = [InferInputConfig(**cfg)for cfg in self.infer_inputs]
        self.train_inputs = [TrainInputConfig(**cfg)for cfg in self.train_inputs]

@dataclass
class Frontend:
    name: str # 前端模型类型

    def load(self, path, shape_dict=None, **kwargs):
        if self.name == "caffe":
            caffe = lazy_import("caffe")
            text_format = lazy_import(".text_format", "google.protobuf")
            pb = lazy_import(".caffe_pb2", "caffe.proto")
            init_net = pb.NetParameter()
            predict_net = pb.NetParameter()
            caffe.set_mode_cpu() # CPU 模式
            # caffe_pb2.NetParameter，模型权重文件
            with open(path, 'rb') as f:
                init_net.ParseFromString(f.read())
            # caffe_pb2.NetParameter，caffe prototxt
            with open(kwargs["predict_net_path"], 'r') as f:
                text_format.Merge(f.read(), predict_net)
            dtype_dict = kwargs["dtype_dict"]
            mod, params = relay.frontend.from_caffe(init_net, predict_net, shape_dict, dtype_dict)
            return TVMCModel(mod, params)
        elif self.name in ["keras", "onnx", "pb", "tflite", "pytorch", "paddle", "relay"]:
            return load_model(path, self.name, shape_dict=shape_dict, **kwargs)
        else:
            raise TypeError(f"暂未支持 {self.name} 前端")
