..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

集成 TVM 到你的项目
===============================

TVM 的运行时被设计为轻量级和可移植的。有几种方法可以将 TVM 集成到项目中。

本文介绍了将 TVM 集成为 JIT 编译器以在系统上生成函数的可能方法。

支持 DLPack
--------------

TVM 生成的函数遵循 PackedFunc 协议。它是可以接受位置参数的函数，包括标准类型，如浮点数、整数、字符串。

PackedFunc 接受 `DLPack <https://github.com/dmlc/dlpack>`_  协议的 DLTensor 指针。所以你唯一需要解决的事情就是创建对应的 DLTensor 对象。

集成用户定义的 C++ Array
--------------------------------

在 C++ 中唯一要做的事情就是将数组转换为 DLTensor，并将其地址作为 ``DLTensor*`` 传递给生成的函数。

集成用户定义的 Python Array
-----------------------------------

假设你有 python 对象 ``MyArray``。你需要做三件事

- 添加 ``_tvm_tcode`` 字段到返回 ``tvm.TypeCode.ARRAY_HANDLE`` 的数组中
- 支持在对象中使用 ``_tvm_handle`` 属性，该属性以 Python 整数形式返回 `DLTensor` 的地址
- 通过 ``tvm.register_extension`` 注册这个类

.. code:: python

   # 示例代码
   import tvm

   class MyArray:
       _tvm_tcode = tvm.TypeCode.ARRAY_HANDLE

       @property
       def _tvm_handle(self):
           # 返回 dltensor 地址
           return self.get_dltensor_addr()

   # 可以将注册步骤放在单独的文件 mypkg.tvm.py 中，
   # 如果只想要可选的依赖项，则只能导入该文件
   tvm.register_extension(MyArray)
