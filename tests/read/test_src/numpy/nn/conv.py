from typing import NamedTuple
import numpy as np
from .padding import pad2d


def conv_out_size(n, k, p, s):
    """根据给定的输入大小计算输出大小，
    
    Args:
        n: 宽或高
        k: kernel 大小
        p: 填充
        s: 步幅

    Returns:
        输出大小(宽度或高度)
    """
    return (n - k + 2 * p)//s + 1


class Conv2d(NamedTuple):
    """根据 packing 推导形状

    Args:
        batch: 数据量
        height: 高度
        width: 宽度
        in_channels: 输入通道
        out_channels: 输出通道
        hkernel: 卷积核高度
        wkernel: 卷积核宽度
        hpad: 填充高度
        wpad: 填充宽度
        hstride: 步幅高度
        wstride: 步幅宽度
        groups: 组卷积个数。默认 1。暂未实现组卷积
        dilation: 空洞卷积参数。默认 (1, 1)。暂时不支持空洞卷积
    """
    batch: int
    height: int
    width: int
    in_channels: int
    out_channels: int
    hkernel: int
    wkernel: int
    hpad: int
    wpad: int
    hstride: int
    wstride: int
    groups: int = 1  # 暂未实现组卷积
    dilation: tuple[int, int] = (1, 1)  # 暂时不支持空洞卷积

    @property
    def fout_height(self):
        """输出高度"""
        return conv_out_size(self.height, self.hkernel, self.hpad, self.hstride)

    @property
    def fout_width(self):
        """输出宽度"""
        return conv_out_size(self.width, self.wkernel, self.wpad, self.wstride)

    @property
    def fout_shape(self):
        """输出形状"""
        return self.batch, self.out_channels, self.fout_height, self.fout_width

    def gen_data(self, data_dtype="float32",
                 kernel_dtype="float32"):
        """模拟卷积的参数"""
        data_np = np.random.normal(size=(self.batch, self.in_channels,
                                   self.height, self.width)).astype(data_dtype)
        kernel_np = np.random.normal(size=(self.out_channels, self.in_channels,
                                     self.hkernel, self.wkernel)).astype(kernel_dtype)
        return data_np, kernel_np

    def __call__(self, data_np, kernel_np, out_dtype="float32"):
        """计算卷积"""
        ref_np = np.empty(self.fout_shape, dtype=out_dtype)
        pad_data_np = pad2d(data_np, self.hpad, self.wpad)
        for b in range(self.batch):
            for c_o in range(self.out_channels):
                for i in range(self.fout_height):
                    for j in range(self.fout_width):
                        ref_np[b, c_o, i, j] = 0  # 初始化
                        for c_i in range(self.in_channels):
                            for h_k in range(self.hkernel):
                                for w_k in range(self.wkernel):
                                    _x = pad_data_np[b, c_i, i *
                                                     self.hstride + h_k, j*self.wstride+w_k]
                                    ref_np[b, c_o, i, j] += kernel_np[c_o,
                                                                      c_i, h_k, w_k] * _x
        return ref_np
