#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <stdint.h>

namespace {
void NotLinked() { TVM_FFI_THROW(RuntimeError) << "VTA runtime not linked in flexloopy build"; }
}

namespace vta_rt {
namespace ffi = tvm::ffi;

int64_t command_handle() { return 0; }

void load_2d(int64_t cmd,
             ffi::TensorView src_dram,
             int64_t src_elem_offset,
             int64_t x_size,
             int64_t y_size,
             int64_t x_stride,
             int64_t x_pad_before,
             int64_t y_pad_before,
             int64_t x_pad_after,
             int64_t y_pad_after,
             int64_t dst_sram_index,
             int64_t dst_memory_type) {
  NotLinked();
}

void store_2d(int64_t cmd,
              int64_t src_sram_index,
              int64_t src_memory_type,
              ffi::TensorView dst_dram,
              int64_t dst_elem_offset,
              int64_t x_size,
              int64_t y_size,
              int64_t x_stride) {
  NotLinked();
}

void uop_push(uint32_t mode, uint32_t reset_out, uint32_t dst_index, uint32_t src_index,
              uint32_t wgt_index, uint32_t opcode, uint32_t use_imm, int32_t imm_val) {
  NotLinked();
}

void uop_loop_begin(uint32_t extent, uint32_t dst_factor, uint32_t src_factor,
                    uint32_t wgt_factor) {
  NotLinked();
}

void uop_loop_end() { NotLinked(); }

int dep_push(int64_t, int, int) { NotLinked(); return -1; }

int dep_pop(int64_t, int, int) { NotLinked(); return -1; }

void synchronize(int64_t, int64_t) { NotLinked(); }

int64_t buffer_alloc(int64_t) { NotLinked(); return 0; }

void buffer_free(int64_t) { NotLinked(); }

void buffer_copy(int64_t, int64_t, int64_t, int64_t, int64_t, int) { NotLinked(); }

int64_t buffer_cpu_ptr(int64_t, int64_t) { NotLinked(); return 0; }

void write_barrier(int64_t, int64_t, int64_t, int64_t, int64_t) { NotLinked(); }

void read_barrier(int64_t, int64_t, int64_t, int64_t, int64_t) { NotLinked(); }

TVM_FFI_DLL_EXPORT_TYPED_FUNC(command_handle, vta_rt::command_handle);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(load_2d, vta_rt::load_2d);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(store_2d, vta_rt::store_2d);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(uop_push, vta_rt::uop_push);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(uop_loop_begin, vta_rt::uop_loop_begin);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(uop_loop_end, vta_rt::uop_loop_end);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dep_push, vta_rt::dep_push);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dep_pop, vta_rt::dep_pop);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(synchronize, vta_rt::synchronize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(buffer_alloc, vta_rt::buffer_alloc);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(buffer_free, vta_rt::buffer_free);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(buffer_copy, vta_rt::buffer_copy);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(buffer_cpu_ptr, vta_rt::buffer_cpu_ptr);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(write_barrier, vta_rt::write_barrier);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(read_barrier, vta_rt::read_barrier);

}