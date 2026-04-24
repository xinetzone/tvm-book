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
"""VTA运行时函数的直接FFI绑定层。

本模块为所有VTA硬件操作提供直接的Python绑定，每个函数简单转发调用到底层C库。
所有函数都通过LIB对象调用对应的C函数，LIB对象是在上级模块中已加载的动态库。

VTA（Versatile Tensor Accelerator）是TVM定义的可编程深度学习加速器架构，
这些函数对应VTA指令集的各个操作。
"""

# 从上一级_ffi_api模块导入已加载的动态库对象
from .._ffi_api import LIB


def command_handle():
    """获取VTA命令队列句柄。

    Returns:
        VTA命令队列的句柄，用于后续所有命令操作。
    """
    return LIB.command_handle()


def load_2d(cmd, src_dram, src_elem_offset, x_size, y_size, x_stride,
            x_pad_before, y_pad_before, x_pad_after, y_pad_after,
            dst_sram_index, dst_memory_type):
    """从DRAM加载二维数据到VTA SRAM，支持padding处理。

    Args:
        cmd: VTA命令队列句柄
        src_dram: 源DRAM缓冲区对象
        src_elem_offset: 源数据在DRAM中的元素偏移量（从该位置开始读取）
        x_size: X方向（宽度）的元素个数
        y_size: Y方向（高度）的元素个数
        x_stride: 源数据在X方向上的步长（连续两个元素之间的间距）
        x_pad_before: X方向开头填充0的个数
        y_pad_before: Y方向开头填充0的个数
        x_pad_after: X方向末尾填充0的个数
        y_pad_after: Y方向末尾填充0的个数
        dst_sram_index: 目标SRAM的索引（SRAM银行编号）
        dst_memory_type: 目标内存类型，指示这是输入/权重/输出

    Returns:
        操作结果，通常为指令句柄或状态码。

    Note:
        加载完成后，带padding的二维数据存储在SRAM中，可以被VTA计算单元访问。
    """
    return LIB.load_2d(cmd, src_dram, src_elem_offset, x_size, y_size, x_stride,
                       x_pad_before, y_pad_before, x_pad_after, y_pad_after,
                       dst_sram_index, dst_memory_type)


def store_2d(cmd, src_sram_index, src_memory_type, dst_dram,
             dst_elem_offset, x_size, y_size, x_stride):
    """将VTA SRAM中的二维数据存储回DRAM。

    Args:
        cmd: VTA命令队列句柄
        src_sram_index: 源SRAM的索引（SRAM银行编号）
        src_memory_type: 源内存类型，指示这是输入/权重/输出
        dst_dram: 目标DRAM缓冲区对象
        dst_elem_offset: 目标在DRAM中的元素偏移量（从该位置开始写入）
        x_size: X方向（宽度）的元素个数
        y_size: Y方向（高度）的元素个数
        x_stride: 目标数据在X方向上的步长

    Returns:
        操作结果，通常为指令句柄或状态码。
    """
    return LIB.store_2d(cmd, src_sram_index, src_memory_type, dst_dram,
                        dst_elem_offset, x_size, y_size, x_stride)


def uop_push(mode, reset_out, dst_index, src_index, wgt_index, opcode, use_imm, imm_val):
    """推送一条微操作（micro-op）指令到VTA指令队列。

    Args:
        mode: 操作模式，指示计算模式（如GEMM/ALU等）
        reset_out: 是否重置目的寄存器（0不重置，1重置）
        dst_index: 目的寄存器索引
        src_index: 源寄存器索引
        wgt_index: 权重寄存器索引
        opcode: 操作码，指示具体执行什么运算
        use_imm: 是否使用立即数（0不使用，1使用）
        imm_val: 立即数值，仅当use_imm=1时有效

    Returns:
        操作结果，通常为指令句柄或状态码。

    Note:
        微操作是VTA执行计算的基本单位，一条微操作对应一次乘加计算或ALU操作。
    """
    return LIB.uop_push(mode, reset_out, dst_index, src_index, wgt_index, opcode, use_imm, imm_val)


def uop_loop_begin(extent, dst_factor, src_factor, wgt_factor):
    """标记微操作循环开始，设置循环参数。

    Args:
        extent: 循环迭代总次数
        dst_factor: 目的维度的循环因子（每次迭代处理多少个目的元素）
        src_factor: 源维度的循环因子（每次迭代处理多少个源元素）
        wgt_factor: 权重维度的循环因子（每次迭代处理多少个权重元素）

    Returns:
        循环句柄，用于匹配对应的uop_loop_end。

    Note:
        通过嵌套uop_loop_begin/uop_loop_end可以实现多重循环，
        用于展开大尺寸张量计算到VTA有限的SRAM容量。
    """
    return LIB.uop_loop_begin(extent, dst_factor, src_factor, wgt_factor)


def uop_loop_end():
    """标记微操作循环结束。

    Returns:
        操作结果，通常为状态码。

    Note:
        必须与前面的uop_loop_begin配对使用，表示循环体结束。
    """
    return LIB.uop_loop_end()


def dep_push(cmd, from_qid, to_qid):
    """推送依赖关系，建立从一个队列到另一个队列的依赖。

    Args:
        cmd: VTA命令队列句柄
        from_qid: 源队列ID，依赖的起点
        to_qid: 目标队列ID，依赖的终点

    Returns:
        操作结果，通常为状态码。

    Note:
        用于保证指令执行顺序：to_qid的指令必须等待from_qid的指令完成后才能执行。
        这用于管理不同硬件单元之间的数据依赖。
    """
    return LIB.dep_push(cmd, from_qid, to_qid)


def dep_pop(cmd, from_qid, to_qid):
    """弹出依赖关系，移除从一个队列到另一个队列的依赖。

    Args:
        cmd: VTA命令队列句柄
        from_qid: 源队列ID
        to_qid: 目标队列ID

    Returns:
        操作结果，通常为状态码。

    Note:
        与dep_push配对使用，依赖关系完成后弹出，可用于下一次依赖建立。
    """
    return LIB.dep_pop(cmd, from_qid, to_qid)


def synchronize(cmd, wait_cycles):
    """同步等待指定数量的时钟周期。

    Args:
        cmd: VTA命令队列句柄
        wait_cycles: 需要等待的时钟周期数

    Returns:
        操作结果，通常为状态码。

    Note:
        用于在指令流中插入延迟，保证数据就绪或解决时序问题。
    """
    return LIB.synchronize(cmd, wait_cycles)


def buffer_alloc(nbytes):
    """分配指定大小的VTA设备缓冲区。

    Args:
        nbytes: 需要分配的字节数

    Returns:
        分配成功返回缓冲区指针，失败返回NULL。

    Note:
        分配的内存是VTA设备可访问的内存空间，可能是DRAM也可能是SRAM，
        取决于具体的VTA硬件实现。
    """
    return LIB.buffer_alloc(nbytes)


def buffer_free(ptr):
    """释放先前分配的VTA设备缓冲区。

    Args:
        ptr: 先前buffer_alloc返回的缓冲区指针

    Returns:
        操作结果，通常为状态码。

    Warning:
        必须配对使用，避免内存泄漏。释放后不能再访问该缓冲区。
    """
    return LIB.buffer_free(ptr)


def buffer_copy(from_ptr, from_offset, to_ptr, to_offset, size, kind_mask):
    """在缓冲区之间拷贝数据。

    Args:
        from_ptr: 源缓冲区指针
        from_offset: 源缓冲区偏移量（字节）
        to_ptr: 目标缓冲区指针
        to_offset: 目标缓冲区偏移量（字节）
        size: 需要拷贝的字节数
        kind_mask: 类型掩码，指示拷贝方向（如CPU→VTA，VTA→CPU，VTA→VTA等）

    Returns:
        操作结果，通常为状态码。

    Note:
        支持不同内存空间之间的数据拷贝，用于在CPU和VTA之间传输数据。
    """
    return LIB.buffer_copy(from_ptr, from_offset, to_ptr, to_offset, size, kind_mask)


def buffer_cpu_ptr(cmd, buffer):
    """获取缓冲区的CPU可访问指针。

    Args:
        cmd: VTA命令队列句柄
        buffer: VTA缓冲区对象

    Returns:
        CPU地址空间中的指针，可以直接读写缓冲区内容。

    Note:
        VTA缓冲区对象可能封装了设备地址，本函数将其转换为CPU可以直接
        访问的虚拟地址，用于CPU初始化数据或读取计算结果。
    """
    return LIB.buffer_cpu_ptr(cmd, buffer)


def write_barrier(cmd, buffer, elem_bits, start, extent):
    """写入内存屏障，保证之前的写入操作对所有读取者可见。

    Args:
        cmd: VTA命令队列句柄
        buffer: 需要屏障的缓冲区对象
        elem_bits: 每个元素的位宽（如8bit、16bit、32bit）
        start: 屏障覆盖的起始元素索引
        extent: 屏障覆盖的元素个数

    Returns:
        操作结果，通常为状态码。

    Note:
        写屏障保证在屏障之前的所有写操作都完成并可见后，
        后续的读操作才能开始执行，防止指令重排导致数据不一致。
    """
    return LIB.write_barrier(cmd, buffer, elem_bits, start, extent)


def read_barrier(cmd, buffer, elem_bits, start, extent):
    """读取内存屏障，保证之前的读取操作完成。

    Args:
        cmd: VTA命令队列句柄
        buffer: 需要屏障的缓冲区对象
        elem_bits: 每个元素的位宽（如8bit、16bit、32bit）
        start: 屏障覆盖的起始元素索引
        extent: 屏障覆盖的元素个数

    Returns:
        操作结果，通常为状态码。

    Note:
        读屏障保证在屏障之前的所有读操作都完成后，
        后续的写操作才能开始执行，防止数据竞争问题。
    """
    return LIB.read_barrier(cmd, buffer, elem_bits, start, extent)