from .._ffi_api import LIB


def command_handle():
    return LIB.command_handle()


def load_2d(cmd, src_dram, src_elem_offset, x_size, y_size, x_stride,
            x_pad_before, y_pad_before, x_pad_after, y_pad_after,
            dst_sram_index, dst_memory_type):
    return LIB.load_2d(cmd, src_dram, src_elem_offset, x_size, y_size, x_stride,
                       x_pad_before, y_pad_before, x_pad_after, y_pad_after,
                       dst_sram_index, dst_memory_type)


def store_2d(cmd, src_sram_index, src_memory_type, dst_dram,
             dst_elem_offset, x_size, y_size, x_stride):
    return LIB.store_2d(cmd, src_sram_index, src_memory_type, dst_dram,
                        dst_elem_offset, x_size, y_size, x_stride)


def uop_push(mode, reset_out, dst_index, src_index, wgt_index, opcode, use_imm, imm_val):
    return LIB.uop_push(mode, reset_out, dst_index, src_index, wgt_index, opcode, use_imm, imm_val)


def uop_loop_begin(extent, dst_factor, src_factor, wgt_factor):
    return LIB.uop_loop_begin(extent, dst_factor, src_factor, wgt_factor)


def uop_loop_end():
    return LIB.uop_loop_end()


def dep_push(cmd, from_qid, to_qid):
    return LIB.dep_push(cmd, from_qid, to_qid)


def dep_pop(cmd, from_qid, to_qid):
    return LIB.dep_pop(cmd, from_qid, to_qid)


def synchronize(cmd, wait_cycles):
    return LIB.synchronize(cmd, wait_cycles)


def buffer_alloc(nbytes):
    return LIB.buffer_alloc(nbytes)


def buffer_free(ptr):
    return LIB.buffer_free(ptr)


def buffer_copy(from_ptr, from_offset, to_ptr, to_offset, size, kind_mask):
    return LIB.buffer_copy(from_ptr, from_offset, to_ptr, to_offset, size, kind_mask)


def buffer_cpu_ptr(cmd, buffer):
    return LIB.buffer_cpu_ptr(cmd, buffer)


def write_barrier(cmd, buffer, elem_bits, start, extent):
    return LIB.write_barrier(cmd, buffer, elem_bits, start, extent)


def read_barrier(cmd, buffer, elem_bits, start, extent):
    return LIB.read_barrier(cmd, buffer, elem_bits, start, extent)