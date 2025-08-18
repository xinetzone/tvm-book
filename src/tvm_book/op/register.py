"""算子注册模块

此模块提供了算子注册相关的功能，包括算子定义、模式注册、状态标记和分区函数注册等。

Examples:
    >>> from tvm_book.op import register
    >>> reg = register.Register("my_op")
    >>> reg.op("My custom operator")
    >>> reg.pattern(register.OpPattern.ELEMWISE)
"""
from dataclasses import dataclass
from typing import Optional, Callable
import tvm
# from tvm.ir import Op
from .attr_types import OpPatternKind


@dataclass
class Register:
    """算子注册类

    提供算子注册相关的方法，包括获取或创建算子、注册算子模式、标记算子状态等。

    Attributes:
        op_name: str
            算子的名称
    """
    op_name: str  # 算子的名称

    def describe(self, info: Optional[str] = "") -> None:
        """获取或创建指定名称的算子

        如果算子未注册，则创建一个新的空算子；如果已注册，则抛出错误。

        Args:
            info: 算子的描述信息
        """
        tvm.ir._ffi_api.RegisterOp(self.op_name, info)

    def pattern(self, pattern: OpPatternKind, level: int = 10) -> Callable:
        """注册算子的模式

        为指定的算子注册一个模式，用于优化和融合决策。

        Args:
            pattern: 使用的模式类型
            level: 优先级级别，默认为10
        Returns:
            Callable
                装饰器函数
        """
        return tvm.ir.register_op_attr(self.op_name, "TOpPattern", pattern, level)

    def stateful(self, stateful: bool, level: int = 10) -> None:
        """注册算子的有状态标志

        标记算子是否为有状态的（即输出依赖于先前的输入或内部状态）。

        Args:
            stateful: 有状态标志，True表示有状态，False表示无状态
            level: 优先级级别，默认为10
        """
        tvm.ir.register_op_attr(self.op_name, "TOpIsStateful", stateful, level)

    def partition_function(self, frewrite: Optional[Callable] = None, level: int = 10) -> Callable:
        """注册算子的分区重写函数

        为指定的算子注册一个分区重写函数，用于在量化过程中确定如何对该算子进行分区。

        Args:
            frewrite: 分区重写函数，接受(ref_call, new_args, ctx)参数，默认为None
            level: 注册级别，默认为10

        Returns:
            Callable
                装饰器函数
        """
        return tvm.ir.register_op_attr(self.op_name, "FQPartitionRewrite", frewrite, level)
