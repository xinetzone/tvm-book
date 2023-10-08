/*!
 * \brief TVM 运行时打包到一个文件
 *
 *   您只需使用此文件编译 libtvm_runtime 以包含在你的项目。
 * 
 *  - 将此文件复制到依赖于 tvm 运行时的项目中。
 *  - 使用 -std=c++17 编译
 *  - 添加以下 include 路径
 *     - /path/to/tvm/include/
 *     - /path/to/tvm/3rdparty/dmlc-core/include/
 *     - /path/to/tvm/3rdparty/dlpack/include/
 *   - 将 -lpthread -ldl 添加到链接库。
 *
 * 这里的包含文件是用相对路径表示的
 * 您需要记住更改它以指向正确的文件。
 *
 * \file tvm_runtime_pack.cc
 */
// const TVM_USE_LIBBACKTRACE = 0 // 替换
#define TVM_USE_LIBBACKTRACE 0
#include "../../../src/runtime/c_runtime_api.cc"
#include "../../../src/runtime/container.cc"
#include "../../../src/runtime/cpu_device_api.cc"
#include "../../../src/runtime/file_utils.cc"
#include "../../../src/runtime/library_module.cc"
#include "../../../src/runtime/logging.cc"
#include "../../../src/runtime/module.cc"
#include "../../../src/runtime/ndarray.cc"
#include "../../../src/runtime/object.cc"
#include "../../../src/runtime/registry.cc"
#include "../../../src/runtime/thread_pool.cc"
#include "../../../src/runtime/threading_backend.cc"
#include "../../../src/runtime/workspace_pool.cc"

// 注意：在此之后的所有文件都是可选模块，
// 你可以包括删除，这取决于你使用多少功能。

// 很可能只需要启用以下其中之一
// 如果使用 Module::Load，则使用 dso_module
// 对于系统打包库，使用 system_lib_module 
#include "../../../src/runtime/dso_library.cc"
// #include "../../../src/runtime/system_library.cc"

// Graph executor
#include "../../../src/runtime/graph_executor/graph_executor.cc"
#include "../../../src/runtime/graph_executor/graph_executor_factory.cc"

// Graph Debug
// #include "../../../src/runtime/graph_executor/debug/graph_executor_debug.cc"

// PAPI profiling
// #include "../../../src/runtime/profiling.cc"

// 取消注释以下行以启用 RPC
// #include "../../../src/runtime/rpc/rpc_session.cc"
// #include "../../../src/runtime/rpc/rpc_event_impl.cc"
// #include "../../../src/runtime/rpc/rpc_server_env.cc"

// 这些宏在未注释时启用设备 API。
// #define TVM_CUDA_RUNTIME 1
// #define TVM_METAL_RUNTIME 1
// #define TVM_OPENCL_RUNTIME 1

// 取消注释以下行以启用 Metal
// #include "../../../src/runtime/metal/metal_device_api.mm"
// #include "../../../src/runtime/metal/metal_module.mm"

// 取消注释以下行以启用 CUDA
// #include "../../../src/runtime/cuda/cuda_device_api.cc"
// #include "../../../src/runtime/cuda/cuda_module.cc"

// 取消注释以下行以启用 OpenCL
// #include "../../../src/runtime/opencl/opencl_device_api.cc"
// #include "../../../src/runtime/opencl/opencl_module.cc"

// // VTA
// #include "../../../vta/runtime/device_api.cc"
// #include "../../../vta/runtime/runtime.cc"

// // Fsim driver
// #include "../../../3rdparty/vta-hw/src/sim/sim_driver.cc"
// #include "../../../3rdparty/vta-hw/src/sim/sim_tlpp.cc"
// #include "../../../3rdparty/vta-hw/src/vmem/virtual_memory.cc"
