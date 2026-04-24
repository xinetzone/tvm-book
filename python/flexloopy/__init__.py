# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""flexloopy包的公共Python API接口。

本模块是flexloopy包的入口点，负责导出所有公共API接口，包括：
1. 从_ffi_api模块导入所有由tvm-ffi-stubgen自动生成的FFI绑定
2. 处理版本信息管理
3. 提供带有输入验证的自定义包装函数
"""
from typing import Any

# tvm-ffi-stubgen自动生成的导出代码段开始
# fmt: off
# isort: off
from ._ffi_api import *  # noqa: F403
from ._ffi_api import __all__ as _ffi_api__all__

# 初始化__all__列表（如果不存在）
if "__all__" not in globals():
    __all__ = []

# 将_ffi_api中的所有符号添加到当前模块的导出列表
__all__.extend(_ffi_api__all__)
# isort: on
# fmt: on
# tvm-ffi-stubgen自动生成的导出代码段结束

# 版本信息处理
try:
    # 尝试从自动生成的_version模块导入版本信息
    from ._version import __version__, __version_tuple__  # type: ignore[import-not-found]
except ImportError:
    # 如果_version模块不存在，使用默认开发版本信息
    # 格式：主版本.次版本.修订版-开发版本.最后提交哈希.构建日期
    __version__ = "0.0.0.dev0"
    __version_tuple__ = (0, 0, 0, "dev0", "7d34eb8ab.d20250913")

# 将版本信息添加到导出列表
__all__.extend(["__version__", "__version_tuple__"])

# 带有输入验证的add_one自定义包装函数
def add_one(x: Any, y: Any) -> None:
    """对输入张量的每个元素加1，结果写入输出张量。

    这是一个示例函数，演示如何在Python层为底层C函数添加输入验证。

    Args:
        x: 输入张量。必须是一维张量。
            存储待加1的原始数据。
        y: 输出张量。必须是一维张量。
            存储计算结果，需要与x具有相同形状。

    Returns:
        None

    Raises:
        ValueError: 当输入x或输出y不是一维张量时抛出。

    Note:
        首先验证输入张量维度，确保必须是一维，然后调用底层FFI函数执行实际计算。
    """
    # 如果x具有ndim属性且不是一维，抛出参数错误
    if hasattr(x, "ndim") and getattr(x, "ndim") != 1:
        raise ValueError("x must be 1D tensor")
    # 如果y具有ndim属性且不是一维，抛出参数错误
    if hasattr(y, "ndim") and getattr(y, "ndim") != 1:
        raise ValueError("y must be 1D tensor")
    # 调用底层FFI函数执行实际计算
    return _ffi_api.add_one(x, y)

# 将自定义的add_one函数添加到导出列表（如果不存在）
if "add_one" not in __all__:
    __all__.append("add_one")
