#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TVM算子注册工具模块。

该模块提供了用于注册和管理TVM自定义算子的辅助函数和类，
简化算子的类型关系、计算函数、调度策略和模式等注册流程。
"""
from dataclasses import dataclass, field
from typing import List, Optional, Any, Callable, Dict, Union
import tvm
from tvm import relay
from tvm.ir import Op
import numpy as np


def register_op(op_name: str, describe: str = "") -> Op:
    """获取指定名称的算子。

    当算子未注册时，创建一个具有给定名称的新空算子；
    当算子已注册时，返回已注册的算子。

    Args:
        op_name: 算子的名称
        describe: 算子的描述信息，可选

    Returns:
        注册或获取到的算子对象

    Raises:
        RuntimeError: 当注册算子失败时抛出
    """
    try:
        return tvm.ir._ffi_api.RegisterOp(op_name, describe)
    except Exception as e:
        raise RuntimeError(f"Failed to register operator '{op_name}': {str(e)}") from e

@tvm.target.generic_func
def schedule_special_op(attrs: Any, outs: Union[tvm.te.tensor.Tensor, List[tvm.te.tensor.Tensor]], target: tvm.target.Target) -> tvm.te.Schedule:
    """为特殊算子创建默认调度策略。

    为自定义算子创建基础的调度策略，适用于大多数简单算子。

    Args:
        attrs: 算子的属性
        outs: 算子的输出张量或张量列表
        target: 编译目标

    Returns:
        创建的调度对象
    """
    with target:
        # 确保outs始终是一个列表
        outs_list = [outs] if isinstance(outs, tvm.te.tensor.Tensor) else outs
        # 使用第一个输出张量创建调度
        output = outs_list[0]
        schedule = tvm.te.create_schedule(output.op)
        return schedule

@dataclass
class RegisterOp:
    """TVM算子注册辅助类。

    提供简洁的API来注册和管理TVM自定义算子，包括类型关系、计算函数、调度策略和模式等。

    Attributes:
        op_name: 算子名称
        op: 算子对象，初始化后自动获取或注册
    """
    op_name: str  # 算子名称
    op: Op = field(init=False)  # 算子对象，初始化后自动获取或注册

    def __post_init__(self) -> None:
        """初始化后尝试获取或注册算子。

        首先尝试获取已注册的算子，如果不存在则注册新算子。

        Raises:
            RuntimeError: 当获取或注册算子失败时抛出
        """
        try:
            # 尝试获取已注册的算子
            self.op = tvm.ir.Op.get(self.op_name)
        except Exception as e:
            try:
                # 若算子不存在，则注册新算子
                register_op(self.op_name)
                self.op = tvm.ir.Op.get(self.op_name)
            except Exception as inner_e:
                raise RuntimeError(f"Failed to initialize operator '{self.op_name}': {str(inner_e)}") from e

    def add_type_rel(self, rel_name: str, type_rel_func: Optional[Callable] = None) -> None:
        """添加算子的类型关系函数。

        类型关系函数用于推断算子输出的类型信息。

        Args:
            rel_name: 类型关系函数的名称
            type_rel_func: 类型关系函数，可选

        Raises:
            RuntimeError: 当添加类型关系函数失败时抛出
        """
        try:
            self.op.add_type_rel(rel_name, type_rel_func)
        except Exception as e:
            raise RuntimeError(f"Failed to add type relation '{rel_name}' to operator '{self.op_name}': {str(e)}") from e

    def register_compute(self, compute: Optional[Callable] = None, level: int = 10) -> Callable:
        """注册算子的计算函数。

        计算函数定义了算子的计算逻辑。

        Args:
            compute: 计算函数，可选
            level: 注册级别，默认为10

        Returns:
            注册的计算函数装饰器

        Raises:
            RuntimeError: 当注册计算函数失败时抛出
        """
        try:
            return tvm.relay.op.register_compute(self.op_name, compute, level=level)
        except Exception as e:
            raise RuntimeError(f"Failed to register compute function for operator '{self.op_name}': {str(e)}") from e

    def register_schedule(self) -> Callable:
        """注册算子的调度策略。

        使用预定义的schedule_special_op函数作为调度策略。

        Returns:
            注册的调度函数装饰器

        Raises:
            RuntimeError: 当注册调度策略失败时抛出
        """
        try:
            return tvm.relay.op.op.register_schedule(self.op_name, schedule_special_op)
        except Exception as e:
            raise RuntimeError(f"Failed to register schedule for operator '{self.op_name}': {str(e)}") from e

    def register_pattern(self, pattern: int, level: int = 10) -> None:
        """注册算子的模式。

        算子模式用于优化和融合策略选择。

        Args:
            pattern: 算子模式，通常使用tvm.relay.op.OpPattern中的常量
            level: 注册级别，默认为10

        Raises:
            RuntimeError: 当注册算子模式失败时抛出
        """
        try:
            tvm.ir.register_op_attr(self.op_name, "TOpPattern", pattern, level)
        except Exception as e:
            raise RuntimeError(f"Failed to register pattern for operator '{self.op_name}': {str(e)}") from e

    def __call__(self, args: List[Any], attrs: Optional[Dict[str, Any]] = None, 
                type_args: Optional[List[Any]] = None, span: Optional[Any] = None) -> relay.Call:
        """用户友好的回调API接口。

        允许直接调用算子对象创建计算图节点。

        Args:
            args: 算子的输入参数列表
            attrs: 算子的属性字典，可选
            type_args: 算子的类型参数列表，可选
            span: 算子的源位置信息，可选

        Returns:
            创建的relay.Call节点

        Raises:
            RuntimeError: 当调用算子失败时抛出
        """
        try:
            # 处理numpy数组参数
            processed_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    # 将numpy数组转换为TVM常量
                    processed_args.append(tvm.relay.const(arg))
                else:
                    processed_args.append(arg)
            
            return tvm.relay.Call(self.op, processed_args, attrs=attrs, type_args=type_args, span=span)
        except Exception as e:
            raise RuntimeError(f"Failed to call operator '{self.op_name}': {str(e)}") from e

    def register(self, pattern: int, rel_name: str, type_rel_func: Optional[Callable] = None, 
                compute: Optional[Callable] = None, level: int = 10) -> None:
        """一步完成算子的全部注册流程。

        依次注册算子的类型关系、计算函数、调度策略和模式。

        Args:
            pattern: 算子模式，通常使用tvm.relay.op.OpPattern中的常量
            rel_name: 类型关系函数的名称
            type_rel_func: 类型关系函数，可选
            compute: 计算函数，可选
            level: 注册级别，默认为10
        """
        # 注册类型关系
        self.add_type_rel(rel_name, type_rel_func)
        # 注册计算函数
        self.register_compute(compute, level=level)
        # 注册调度策略
        self.register_schedule()
        # 注册算子模式
        self.register_pattern(pattern, level=level)

def _test_register_op() -> None:
    """测试RegisterOp类的功能。

    创建示例算子并验证其注册和执行功能。
    """
    # 创建测试数据
    np_data = np.random.randn(10).astype("float32")  # 随机测试数据
    op_name = "my.operator2"  # 测试算子名称
    
    # 定义类型关系函数
    def my_op_type_rel(arg_types: List[Any], attrs: Any) -> tvm.relay.TensorType:
        """示例算子的类型关系函数。

        定义输入输出类型的关系，这里输出类型与输入类型相同。
        
        Args:
            arg_types: 输入参数类型列表
            attrs: 算子属性
            
        Returns:
            输出张量的类型
        """
        input_type = arg_types[0]
        # 输出类型与输入类型相同
        return tvm.relay.TensorType(input_type.shape, input_type.dtype)
    
    # 定义计算函数
    def my_op_compute(attrs: Any, inputs: List[tvm.te.tensor.Tensor], 
                      output_type: tvm.relay.TensorType) -> List[tvm.te.tensor.Tensor]:
        """示例算子的计算函数。

        实现简单的元素级加法操作：x + 2。
        
        Args:
            attrs: 算子属性
            inputs: 输入张量列表
            output_type: 输出张量类型
            
        Returns:
            输出张量列表
        """
        x = inputs[0]
        # 简单的计算：x + 2
        out = x + 2
        return [out]
    
    try:
        # 创建并注册算子
        op = RegisterOp(op_name)
        pattern = tvm.relay.op.OpPattern.ELEMWISE  # 元素级算子模式
        rel_name = "MyOpTypeRel"  # 类型关系名称
        
        # 执行注册
        op.register(pattern, rel_name, my_op_type_rel, my_op_compute, level=10)
        
        # 创建计算图
        x = tvm.relay.var("x", shape=(10,), dtype="float32")  # 输入变量
        y = op([x])  # 调用算子
        mod = tvm.IRModule.from_expr(tvm.relay.Function([x], y))  # 创建IR模块
        
        # 执行计算
        intrp = tvm.relay.create_executor(kind="vm", mod=mod, device=tvm.cpu(0))
        result = intrp.evaluate()(np_data)
        
        # 验证结果
        expected = np_data + 2  # 算子的预期行为
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)
        print("测试通过!")
    except Exception as e:
        print(f"测试失败: {str(e)}")
        raise


if __name__ == "__main__":
    _test_register_op()
