'''单图处理
'''
import numbers
import numpy as np
from PIL import Image


def resize(img, size, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size, interpolation)


def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    img = img.crop((j, i, j+tw, i+th))
    return img


def mean_std_process(data, mean, std):
    mean = np.array(mean)
    mean = mean[:, np.newaxis, np.newaxis]
    std = np.array(std)
    std = std[:, np.newaxis, np.newaxis]
    data = (data-mean)/std
    return data


def process_single_image(name, mean, std):
    """对图片做预处理"""
    with Image.open(name) as im:
        im = resize(im, (256, 256))
        im = center_crop(im, 224)
        img = np.array(im)
    img = img.transpose((2, 0, 1))
    return mean_std_process(img, mean, std)

def deprocess_single_image(tensor, mean, std):
    """恢复预处理的图片"""
    img = tensor.transpose((1, 2, 0))
    mean = np.array(mean)
    std = np.array(std)
    img = (img * std) + mean
    img = img.clip(0, 255)
    return img.astype("uint8")