from pathlib import Path
import sys
ROOT = Path(__file__).parents[3] # 项目根目录
sys.path.extend([str(ROOT), str(ROOT/"tools_python")])
from dataclasses import dataclass, asdict
import toml

@dataclass
class Config:
    name: str # 输入数据名称
    shape: tuple[int] # 输入数据形状
    model_type: str # 模型所属前端名称，有 'caffe', 'torch', 'tf', 'darknet', 'onnx'
    test_image_path: str # 测试数据路径

configs = [
    ("resnet50_v2", "data", (1, 3, 224, 224), "onnx", "models/resnet50_v2/data/ILSVRC2012_val_00000837.JPEG"),
    ("mobilenet_v2_tf", "data", (1, 3, 224, 224), "onnx", "models/mobilenet_v2_tf/data/ILSVRC2012_val_00000837.JPEG"),

    ("resnet18", "data", (1, 3, 224, 224), "torch", "models/resnet18/data/demo.jpg"),
    ("person", "data", (1, 3, 240, 640), "caffe", "models/person/data/0_cnn.jpg"),
    ("fp_octavia", "data", (1, 3, 112, 112), "caffe", "models/fp_octavia/data/demo.jpg"),
    ("new_person", "data", (1, 3, 240, 640), "caffe", "models/new_person/data/0_cnn.bmp"),
    ("person_car_detect", "input.1", (1, 3, 288, 512), "onnx", "models/person_car_detect/data/demo.jpg"),
    ("person_chair", "input.1", (1, 3, 224, 640), "onnx", "models/person_chair/data/demo.jpg"),
    ("face_classification", "data", (1, 3, 112, 112), "caffe", "models/face_classification/data/cai_lu_yao.jpg"),
    ("face_landmark", "data", (1, 1, 112, 112), "caffe", "models/face_landmark/data/0042_006.jpg"),
    ("face_rec", "data", (1, 3, 112, 112), "caffe", "models/face_rec/test/benxi.jpg"),
    ("fr_karen", "data", (1, 3, 80, 80), "caffe", "models/fr_karen/data/2022714_03821_fa_left.jpg"),
    ("fr_madeline", "input.1", (1, 3, 112, 112), "caffe", "models/fr_madeline/data/benxi.jpg"),
    ("resnet50_v1", "data", (1, 3, 224, 224), "torch", "models/resnet18/data/demo.jpg"),
    ("mobilenet_v2", "data", (1, 3, 224, 224), "torch", "models/resnet18/data/demo.jpg"),
    ("driver", "images", (1, 3, 224, 384), "caffe", "models/driver/data/demo.jpg"),
    ("face_detection_580", "data", (1, 3, 224, 224), "caffe", "models/face_detection_580/data/baibaihe.jpg"),
    ("face_detection", "data", (1, 3, 256, 256), "caffe", "models/face_detection/data/_xinyu_1.jpg"),
    ("fd_quintina", "data", (1, 1, 144, 256), "caffe", "models/fd_quintina/data/demo.jpg"),
    ("yolov5", "images", (1, 3, 640, 640), "onnx", "models/yolov5/data/demo.jpg"),
]

bunch = {config[0]: asdict(Config(*config[1:])) for config in configs}
with open("./model.toml", "w") as fp:
    config = toml.dump(bunch, fp)
