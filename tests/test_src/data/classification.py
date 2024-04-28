from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import numpy as np
import warnings

@dataclass
class ImageFolderDataset:
    """A dataset for loading image files stored in a folder structure.

    like::

        root/car/0001.jpg
        root/car/xxxa.jpg
        root/car/yyyb.jpg
        root/bus/123.jpg
        root/bus/023.jpg
        root/bus/wwww.jpg

    Parameters
    ----------
    root : str
        Path to root directory.
    flag : {0, 1}, default 1
        If 0, always convert loaded images to greyscale (1 channel).
        If 1, always convert loaded images to colored (3 channels).

    Attributes
    ----------
    synsets : list
        List of class names. `synsets[i]` is the name for the integer label `i`
    items : list of tuples
        List of all images in (filename, label) pairs.
    """
    root: str|Path
    flag: int = 1
    
    def __post_init__(self):
        self.root = Path(self.root).expanduser().resolve()
        assert self.flag in {0, 1}, "当前仅仅支持 0 与 1"
        self._exts = ['.jpg', '.jpeg', '.png']
        self._list_images()

    def _list_images(self):
        self.synsets = []
        self.items = []
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                warnings.warn(f'忽略不是目录的文件 {folder}', stacklevel=3)
                continue
            label = len(self.synsets)
            self.synsets.append(folder.name)
            for filename in sorted(folder.iterdir()):
                ext = filename.suffix
                if ext.lower() not in self._exts:
                    warnings.warn("忽略 {ext} 类型的 {filename} of type {ext}，仅仅支持 {self._exts}")
                    continue
                self.items.append((str(filename), label))

    def pil2array(self, im: Image):
        """将 PIL.Image 转换为 np.ndarray"""
        if self.flag:
            im = im.convert("RGB")
        else:
            im = im.convert("L")
        return np.array(im)

    def __getitem__(self, index):
        if isinstance(index, slice):
            images = []
            labels = []
            for path, label in self.items[index]:
                with Image.open(path) as im:
                    img = self.pil2array(im)
                images.append(img)
                labels.append(label)
            return images, labels
        else:
            path, label = self.items[index]
            with Image.open(path) as im:
                img = self.pil2array(im)
            return img, label

    def __len__(self):
        return len(self.items)
