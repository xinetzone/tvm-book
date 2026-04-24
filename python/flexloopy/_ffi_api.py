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
# specific language governing permissions and limitations.
"""flexloopy的FFI API绑定层。

本模块由tvm-ffi-stubgen工具自动生成，负责：
1. 导入TVM FFI相关的基础设施
2. 加载编译好的flexloopy动态库
3. 初始化FFI API绑定
4. 注册C++对象到Python运行时
5. 导出所有公共符号供上层模块使用

FFI（Foreign Function Interface）是TVM用于连接Python和C/C++代码的机制。
"""

# tvm-ffi-stubgen自动生成的导入段开始
# fmt: off
# isort: off
from __future__ import annotations
from tvm_ffi import Object as _ffi_Object, init_ffi_api as _FFI_INIT_FUNC, register_object as _FFI_REG_OBJ
from tvm_ffi.libinfo import load_lib_module as _FFI_LOAD_LIB
from typing import TYPE_CHECKING

# TYPE_CHECKING为True时仅导入用于类型注解
if TYPE_CHECKING:
    from tvm_ffi import Object
# isort: on
# fmt: on
# tvm-ffi-stubgen自动生成的导入段结束

# 加载flexloopy动态库
# 参数：模块名"flexloopy"，库名"flexloopy"
# tvm-ffi-stubgen(import-object): tvm_ffi.libinfo.load_lib_module;False;_FFI_LOAD_LIB
LIB = _FFI_LOAD_LIB("flexloopy", "flexloopy")

# tvm-ffi-stubgen自动生成的全局函数绑定开始
# fmt: off
# 初始化flexloopy模块的FFI API，将C函数绑定到Python模块
_FFI_INIT_FUNC("flexloopy", __name__)

# 仅在类型检查阶段提供函数签名
if TYPE_CHECKING:
    def raise_error(_0: str, /) -> None:
        """抛出错误信息。

        Args:
            _0: 错误信息字符串

        Returns:
            None
        """
# fmt: on
# tvm-ffi-stubgen自动生成的全局函数绑定结束

# tvm-ffi-stubgen声明：导入对象注册函数和基类
# tvm-ffi-stubgen(import-object): tvm_ffi.register_object;False;_FFI_REG_OBJ
# tvm-ffi-stubgen(import-object): ffi.Object;False;_ffi_Object

@_FFI_REG_OBJ("flexloopy.IntPair")
class IntPair(_ffi_Object):
    """IntPair对象的FFI绑定。

    这是C++类flexloopy::IntPair在Python层的绑定类。
    IntPair用于存储一对整数值，演示TVM FFI如何绑定自定义C++对象。

    Extends:
        _ffi_Object: TVM FFI对象基类，提供底层对象管理能力

    Attributes:
        a: int - 第一个整数
        b: int - 第二个整数
    """

    # tvm-ffi-stubgen自动生成的对象绑定开始
    # fmt: off
    # 存储两个整数的属性
    a: int
    b: int

    # 仅在类型检查阶段提供方法签名
    # 实际实现由FFI机制自动生成
    if TYPE_CHECKING:
        @staticmethod
        def __c_ffi_init__(_0: int, _1: int, /) -> Object:
            """C FFI构造函数，从两个整数创建IntPair对象。

            Args:
                _0: 第一个整数值a
                _1: 第二个整数值b

            Returns:
                新建的IntPair对象
            """
        @staticmethod
        def static_get_second(_0: IntPair, /) -> int:
            """静态方法：获取IntPair的第二个整数值。

            Args:
                _0: IntPair对象实例

            Returns:
                第二个整数b的值
            """
        def get_first(self, /) -> int:
            """成员方法：获取IntPair的第一个整数值。

            Returns:
                第一个整数a的值
            """
    # fmt: on
    # tvm-ffi-stubgen自动生成的对象绑定结束


# 导出到上层模块的公共符号列表
__all__ = [
    # tvm-ffi-stubgen(begin): __all__
    "LIB",
    # tvm-ffi-stubgen(end)
]
