# vta_ffi 使用指南

本文档介绍如何在本项目中通过 TVM FFI 调用 VTA 相关功能（下文统称为 vta_ffi），包括基本概念、安装配置、示例代码、常见问题与错误处理、最佳实践与性能优化建议。

## 1. 基本介绍与主要功能

- 本仓库未存在名为 `vta_ffi` 的独立模块，实际采用 TVM FFI（Python 包名为 `tvm_ffi`）作为绑定层，并在扩展库 `tvm_book` 中导出与 VTA 运行时相关的全局函数。
- 关键导出位于 `src/vta_rt.cc`，使用 `TVM_FFI_DLL_EXPORT_TYPED_FUNC` 将下列函数注册为全局可调用符号（可在 Python/C++ 侧通过 FFI 获取并调用）：
  - `command_handle`, `load_2d`, `store_2d`
  - `uop_push`, `uop_loop_begin`, `uop_loop_end`
  - `dep_push`, `dep_pop`
  - `synchronize`
  - `buffer_alloc`, `buffer_free`, `buffer_copy`, `buffer_cpu_ptr`
  - `write_barrier`, `read_barrier`
- 代码参考：`d:\AI\client\tvm-book\src\vta_rt.cc:72-86`（导出符号）；占位实现的报错信息位于 `d:\AI\client\tvm-book\src\vta_rt.cc:8`。
- Python 入口与 API 暴露：`tvm_ffi.__init__` 导出 `get_global_func`, `get_global_func_metadata`, `load_module`, `system_lib` 等接口（`d:\AI\client\tvm-book\extensions\tvm-ffi\python\tvm_ffi\__init__.py:38-45`）。

## 2. 安装与配置（Windows）

### 2.1 安装 TVM FFI（tvm_ffi）

- 方案 A：使用预构建包
  - `pip install tvm-ffi`
  - 成功后，Python 侧可直接 `import tvm_ffi` 使用。
- 方案 B：从源码构建
  - 支持通过 CMake 与 Python 辅助脚本定位源码并构建；在生成步骤中开启 `-DTVM_FFI_BUILD_FROM_SOURCE=ON`。
  - 构建逻辑参考：`d:\AI\client\tvm-book\CMakeLists.txt:40-56`。

### 2.2 构建与安装 tvm_book 扩展库

`tvm_book` 负责链接 `tvm_ffi_header` 与 `tvm_ffi_shared` 并导出上述 VTA 相关符号。

- 关键 CMake 目标与链接：`d:\AI\client\tvm-book\CMakeLists.txt:58-66`
- 安装目标：`d:\AI\client\tvm-book\CMakeLists.txt:78`

示例命令（Windows，VS 2022，x64）：

```powershell
# 在仓库根目录 d:\AI\client\tvm-book 下
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DTVM_FFI_BUILD_FROM_SOURCE=ON
cmake --build build --config Release
cmake --install build --config Release --prefix .
```

执行成功后，会在安装前缀路径生成 `tvm_book` 对应的动态库（Windows 下通常为 `tvm_book.dll`）。

### 2.3 Python 环境配置（TVM/VTA）

若需要在同一 Python 环境中直接使用上游 TVM/VTA 的 Python 包，可将其加入 `sys.path` 或设置 `PYTHONPATH`。本仓库提供了一个示例配置（注意其中路径分隔符在 Windows 下建议改为分号 `;`）：

- 参考实现：`d:\AI\client\tvm-book\python\tvm_book\config\env.py:7-22`

示例（PowerShell）：

```powershell
# 假设 TVM 根目录为 D:\projects\tvm
$env:PYTHONPATH = "D:\projects\tvm\python;D:\projects\tvm\vta\python;" + $env:PYTHONPATH
```

## 3. 使用示例与代码片段

### 3.1 Python：获取与调用全局函数

通过 `tvm_ffi.get_global_func` 检索并调用导出的全局函数。以下示例演示获取句柄与加载 2D 数据（加载调用在未链接真实 VTA runtime 时会抛出占位错误，用于说明错误处理流程）。

```python
import numpy as np
import tvm_ffi

# 1) 获取 command_handle 并调用（占位实现返回整型句柄，当前为 0）
f_handle = tvm_ffi.get_global_func("command_handle")
cmd = f_handle()
print("cmd handle:", cmd)  # 期望输出 0（占位实现），真实绑定后为设备/命令句柄

# 2) 准备一个张量并调用 load_2d（示例演示参数排列与错误捕获）
x = np.arange(16, dtype=np.int32).reshape(4, 4)
tensor = tvm_ffi.from_dlpack(x)

f_load_2d = tvm_ffi.get_global_func("load_2d")
try:
    f_load_2d(
        int(cmd),        # cmd
        tensor,          # src_dram: TensorView 兼容对象
        0,               # src_elem_offset
        4, 4,            # x_size, y_size
        4,               # x_stride
        0, 0,            # x_pad_before, y_pad_before
        0, 0,            # x_pad_after, y_pad_after
        0,               # dst_sram_index
        0,               # dst_memory_type
    )
except tvm_ffi.error.Error as e:
    # 当前占位实现会抛出：VTA runtime not linked in tvm_book build
    print("FFI Error:", e)

# 3) 查询函数元数据/签名（若提供）
meta = tvm_ffi.get_global_func_metadata("load_2d")
print("load_2d metadata:", meta)
```

说明：
- `command_handle` 的占位实现返回 `int64` 句柄（`d:\AI\client\tvm-book\src\vta_rt.cc:14`）。
- 其他函数在未链接真实 VTA runtime 时会抛出错误（`d:\AI\client\tvm-book\src\vta_rt.cc:8`）。
- `TensorView` 参数在 Python 侧可通过 `tvm_ffi.Tensor`/`from_dlpack` 进行传递。

### 3.2 Python：通过模块加载扩展库（可选）

`tvm_ffi.load_module("路径")` 可显式加载共享库对象并获取函数（当符号以模块形式导出时尤为便利）。当前导出采用“全局函数”注册，通常直接使用 `get_global_func` 即可。

```python
import tvm_ffi
mod = tvm_ffi.load_module("./tvm_book.dll")  # 具体路径按实际构建位置调整
print("module kind:", mod.kind)
print("has load_2d:", mod.implements_function("load_2d", query_imports=True))
```

### 3.3 C++：从全局函数检索并调用（示意）

```cpp
#include <tvm/ffi/function.h>
#include <tvm/ffi/container/tensor.h>

int main() {
  using tvm::ffi::Function;
  // 获取 command_handle
  Function f_handle = Function::GetGlobalRequired("command_handle");
  int64_t cmd = f_handle();

  // 获取 load_2d（当前占位实现会抛错）
  Function f_load = Function::GetGlobalRequired("load_2d");
  // TODO: 构造 ffi::TensorView 并按需填参后调用
}
```

## 4. 常见问题与错误处理

- 找不到全局函数（`ValueError: Cannot find global function ...`）
  - 原因：共享库未加载/未位于可搜索路径、符号名不匹配、构建失败或未安装。
  - 处理：确认 `tvm_book` 构建产物路径；必要时使用 `tvm_ffi.load_module("路径")` 显式加载；检查符号名拼写；验证构建日志。
- `VTA runtime not linked in tvm_book build`
  - 原因：`src/vta_rt.cc` 为占位实现，未链接真实 VTA runtime。
  - 处理：将相应函数替换为真实实现或在构建时链接 VTA 运行时库；完成后可在 Python/C++ 侧重试调用。
- 张量/类型不匹配
  - 建议：使用 `tvm_ffi.from_dlpack` 与 `tvm_ffi.dtype(...)` 保证 dtype 与形状一致；注意设备类型（`tvm_ffi.device("cpu", 0)` 等）。
- Windows 环境路径问题
  - 使用分号 `;` 作为 `PYTHONPATH` 分隔符；确保 VS 工具链与 Python 版本 ABI 兼容。

## 5. 最佳实践与性能优化

- 缓冲区管理
  - 结合 `buffer_alloc/free/copy/cpu_ptr`，实现高效的内存管理与数据传输。
- 降低 Python GIL 影响
  - 对计算密集型 FFI 函数，可在 Python 侧设置 `Function.release_gil = True`（具体可用性依实现而定）。
- 零拷贝/内存布局
  - 使用 DLPack 进行零拷贝数据互通，避免多余转换；根据硬件需求选择合适的步幅与对齐策略。
- 模块生命周期
  - 避免在短作用域频繁加载/卸载共享库，保持长生命周期以防对象析构与卸载顺序冲突（参见 `tvm_ffi.Module` 的使用建议，`d:\AI\client\tvm-book\extensions\tvm-ffi\python\tvm_ffi\module.py:77-93`）。
- 同步与屏障
  - 正确地在读写前后使用 `write_barrier`/`read_barrier`，并在必要时调用 `synchronize` 保证一致性。

## 参考文件与定位

- `d:\AI\client\tvm-book\src\vta_rt.cc:8, 14, 72-86`：VTA 相关 FFI 导出与占位实现。
- `d:\AI\client\tvm-book\CMakeLists.txt:40-56, 58-66, 78`：`tvm_ffi` 引入方式与 `tvm_book` 构建安装。
- `d:\AI\client\tvm-book\python\tvm_book\config\env.py:7-22`：Python 环境路径配置示例。
- `d:\AI\client\tvm-book\extensions\tvm-ffi\python\tvm_ffi\__init__.py:38-45`：Python 侧公开 API（`get_global_func` 等）。

如需进一步扩展到真实 VTA 加速器，请将 `src/vta_rt.cc` 中的占位函数替换为实际运行时绑定，并保持导出符号名不变，以便现有 Python/C++ 侧调用无需改动。