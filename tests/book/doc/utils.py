from PIL import Image
import numpy as np

def letterbox_image(im: Image, dst_width: int, dst_height: int):
    '''使用填充保持纵横比缩放图像
    
    Args:
        im: 原始 Image
        dst_width: 目标宽度
        dst_height: 目标高度
    '''
    iw, ih = im.size # 原始尺寸
    scale = min(dst_width/iw, dst_height/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    im = im.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (dst_width, dst_height), (114, 114, 114))
    new_image.paste(im, ((dst_width-nw)//2, (dst_height-nh)//2))
    return new_image

def preprocessing(path: str|None, **ENV: dict):
    if not path:
        im = np.random.randint(0, 256, size=(32, 32, 3), dtype="uint8")
        im = Image.fromarray(im) # 转为 Image 实例
    else:
        im = Image.open(path)
    # im = im.resize((ENV["width"], ENV["height"]), Image.BICUBIC)
    im = letterbox_image(im, ENV["width"], ENV["height"])
    if ENV["mode"] == "L": # 将灰度图转换为 HWC 布局
        img = im.convert("L")
        img = np.expand_dims(img, axis=-1) # 转为 HWC
    elif ENV["mode"] == "RGB":
        img = np.array(im.convert("RGB")) # 转为 HWC 布局
    elif ENV["mode"] == "BGR":
        img = np.array(im.convert("RGB")) # 转为 HWC 布局
        img = img[..., ::-1] # RGB 转 BGR
    else:
        raise TypeError(f'暂未支持数据布局 {ENV["mode"]}')
    image_np = np.expand_dims(img, 0) # 转换为 NHWC (uint8 数据)
    # 预处理后的数据
    data_inp = ((image_np - ENV["mean"]) / ENV["std"]).astype(np.float32)
    data_inp = data_inp.transpose(0, 3, 1, 2)
    return np.ascontiguousarray(image_np), np.ascontiguousarray(data_inp)

