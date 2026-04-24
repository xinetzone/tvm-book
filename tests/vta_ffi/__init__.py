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
"""VTA FFI运行时绑定 - 为flexloopy提供VTA硬件操作接口。

本模块导出所有VTA（Versatile Tensor Accelerator，通用张量加速器）运行时函数，
允许flexloopy通过FFI机制调用VTA硬件指令。VTA是TVM项目中定义的可编程加速器架构，
用于加速深度学习计算。

导出的函数按功能分类：
- 命令处理：command_handle - 获取命令句柄
- 内存读写：load_2d, store_2d - 在DRAM和SRAM之间传输二维数据
- 微操作指令：uop_push, uop_loop_begin, uop_loop_end - 构建微操作指令流
- 依赖管理：dep_push, dep_pop, synchronize - 管理指令间依赖关系
- 缓冲区管理：buffer_alloc, buffer_free, buffer_copy, buffer_cpu_ptr - 缓冲区操作
- 内存屏障：write_barrier, read_barrier - 保证内存可见性
"""

# 从_ffi_api模块导入所有VTA运行时函数并重新导出
from ._ffi_api import (
    command_handle,        # 获取命令句柄
    load_2d,               # 从DRAM加载二维数据到SRAM
    store_2d,              # 将SRAM中的二维数据存储回DRAM
    uop_push,              # 推送一条微操作指令
    uop_loop_begin,        # 标记循环开始
    uop_loop_end,          # 标记循环结束
    dep_push,              # 推送依赖关系
    dep_pop,               # 弹出依赖关系
    synchronize,          # 同步等待指定周期
    buffer_alloc,         # 分配缓冲区
    buffer_free,          # 释放缓冲区
    buffer_copy,          # 缓冲区拷贝
    buffer_cpu_ptr,       # 获取缓冲区的CPU可访问指针
    write_barrier,        # 写内存屏障
    read_barrier,         # 读内存屏障
)