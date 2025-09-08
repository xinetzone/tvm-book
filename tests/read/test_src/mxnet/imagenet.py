from dataclasses import dataclass
from pathlib import Path
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from mxnet.io import ImageRecordIter
from gluoncv.data.imagenet.classification import ImageNet
import numpy as np
from .tools import Estimator

ROOT = "/media/pc/data/4tb/lxw/remote/pc/.mxnet/datasets/imagenet"
normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])


def crop(input_size, crop_ratio=0.875):
    '''仅用于验证数据集，与 TF 实现对齐，设置默认的裁剪输入比为 0.875；

    将裁剪设置为 ``ceil(input-size/ratio)``
    '''
    crop_ratio = crop_ratio if crop_ratio > 0 else 0.875
    # resize 后的尺寸
    return int(np.ceil(input_size/crop_ratio))


@dataclass
class TestsetConfig:
    batch_size: int = 32  # training batch size per device (CPU/GPU).
    input_size: int = 224  # input shape of the image
    num_workers: int = 4


def get_valset(config,
               root: str = ROOT,
               crop_ratio: float = 0.875):
    '''

    Args:
        config: TestsetConfig 实例
    '''
    resize = crop(config.input_size, crop_ratio)  # 与 TF 实现对齐
    transform_test = transforms.Compose([
        transforms.Resize(resize, keep_ratio=True),
        transforms.CenterCrop(config.input_size),
        transforms.ToTensor(),
        normalize
    ])

    imagenet_val = ImageNet(root=root, train=False)
    return gluon.data.DataLoader(
        imagenet_val.transform_first(transform_test),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers)


def get_valsetrec(config,
                  root: str = f'{ROOT}/rec',
                  crop_ratio: float = 0.875):
    '''

    Args:
        config: TestsetConfig 实例
        rec_dir: recio directory for validation.
    '''
    rec_dir = Path(root)
    imgrec = rec_dir/'val.rec'
    imgidx = rec_dir/'val.idx'
    resize = crop(config.input_size, crop_ratio)  # 与 TF 实现对齐
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]
    return ImageRecordIter(
        path_imgrec=imgrec,
        path_imgidx=imgidx,
        shuffle=False,
        preprocess_threads=config.num_workers,
        batch_size=config.batch_size,
        resize=resize,
        data_shape=(3, config.input_size, config.input_size),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
    )


def imagenet_estimator(logger, config, mode='image',
                       crop_ratio: float = 0.875):
    root = ROOT if mode == 'image' else f'{ROOT}/rec'
    kwargs = {
        'config': config,
        'crop_ratio': crop_ratio,
        'root': root
    }
    if mode == 'image':
        val_data = get_valset(**kwargs)
    else:
        val_data = get_valsetrec(**kwargs)
    return Estimator(logger, val_data, mode)
