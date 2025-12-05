from dataclasses import dataclass
from typing import Optional
import tvm
from tvm import DataType, relax, tir
from tvm.relax.testing import nn
from tvm.script import ir as I, relax as R, tir as T
from tvm.relax import op as _op

@dataclass
class Conv2D(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: list[int] | int
    strides: int | tuple[int, int] = 1
    padding: int | tuple[int, ...] = 0
    dilation: int | tuple[int, int] = 1
    groups: int = 1
    bias: bool = True
    data_layout: str = 'NCHW'
    kernel_layout: str = 'OIHW'
    out_layout: Optional[str] = None
    out_dtype: Optional[str | DataType] = None
    name: str = "conv2d"
    define_subroutine: bool = True

    def __post_init__(self):
        # Allow dynamic input channels.
        if isinstance(self.in_channels, int):
            in_channels = int(self.in_channels / self.groups)
        else:
            in_channels = tir.floordiv(self.in_channels, self.groups)

        # Expand kernel size if provided an integer.
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        else:
            self.kernel_size = self.kernel_size

        kernel_shape = [self.out_channels, in_channels] + list(self.kernel_size)

        self.weight = nn.Parameter(kernel_shape, self.out_dtype, name="weight")

        if self.bias:
            self.bias = nn.Parameter((self.out_channels,), self.out_dtype, name="bias")
        else:
            self.bias = None
    
    def forward(self, x: relax.Expr) -> relax.Var:
        conv_out = _op.nn.conv2d(
            data=x,
            weight=self.weight,
            strides=self.strides,
            padding=self.padding,
            dilation=self.dilation,
            data_layout=self.data_layout,
            groups=self.groups,
            kernel_layout=self.kernel_layout,
            out_layout=self.out_layout,
            out_dtype=self.out_dtype,
        )
        if self.bias is not None:
            if self.data_layout == "NCHW":
                conv_out = _op.add(conv_out, _op.reshape(self.bias, [1, -1, 1, 1]))
            elif self.data_layout == "NHWC":
                conv_out = _op.add(conv_out, _op.reshape(self.bias, [1, 1, 1, -1]))
            else:
                raise NotImplementedError(f"Dont know how to handle layout {self.data_layout}.")

        return nn.emit(conv_out, self.name)

@dataclass
class ConvTranspose2D(nn.Module):
    """
    Module for ConvTranspose1D layer.
    """
    in_channels: int
    out_channels: int
    kernel_size: list[int] | int
    strides: int | tuple[int, int] = 1
    padding: int | tuple[int, ...] = 0
    output_padding: int | tuple[int, int] = 0
    dilation: int | tuple[int, int] = 1
    groups: int = 1
    bias: bool = True
    data_layout: str = 'NCHW'
    kernel_layout: str = 'IOHW'
    out_layout: Optional[str] = None
    out_dtype: Optional[str | DataType] = None
    name: str = "conv2d_transpose"
    define_subroutine: bool = True

    def __post_init__(self):
        # Allow dynamic output channels.
        if isinstance(self.in_channels, int):
            out_channels = int(self.out_channels / self.groups)
        else:
            out_channels = tir.floordiv(self.out_channels, self.groups)

        # Expand kernel size if provided an integer.
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        else:
            self.kernel_size = self.kernel_size

        kernel_shape = [self.in_channels, out_channels] + list(self.kernel_size)
        self.weight = nn.Parameter(kernel_shape, self.out_dtype, name="weight")

        if self.bias:
            self.bias = nn.Parameter((self.out_channels,), self.out_dtype, name="bias")
        else:
            self.bias = None

    def forward(self, x: relax.Expr) -> relax.Var:
        out = _op.nn.conv2d_transpose(
            data=x,
            weight=self.weight,
            strides=self.strides,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            data_layout=self.data_layout,
            groups=self.groups,
            kernel_layout=self.kernel_layout,
            out_layout=self.out_layout,
            out_dtype=self.out_dtype,
        )
        if self.bias is not None:
            if self.data_layout == "NCHW":
                out = _op.add(out, _op.reshape(self.bias, [1, -1, 1, 1]))
            elif self.data_layout == "NHWC":
                out = _op.add(out, _op.reshape(self.bias, [1, 1, 1, -1]))
            else:
                raise NotImplementedError(f"Dont know how to handle layout {self.data_layout}.")
        return nn.emit(out, self.name)

        

