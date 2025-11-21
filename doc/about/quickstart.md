# 快速上手

- 安装与构建
  - 安装依赖：`pip install -e .` 或 `pip install tvm-book`
  - 如需从源码构建 FFI：设置 `TVM_FFI_EXT_FROM_SOURCE=ON` 并使用 `pip install -e .`
- 加载共享库
  - 默认自动定位构建产物；可通过环境变量覆盖：
    - `TVM_BOOK_LIB_PATH` 指定完整库路径，如 `d:/build/tvm_book.dll`
    - `TVM_BOOK_LIB_DIR` 指定目录，自动在其中寻找库文件
- 基本用例
  - `add_one`
    ```python
    import numpy as np
    import tvm_book
    x = np.array([1, 2, 3], dtype=np.float32)
    y = np.empty_like(x)
    tvm_book.add_one(x, y)
    assert np.allclose(y, x + 1)
    ```
  - 自定义对象
    ```python
    import tvm_book
    pair = tvm_book.IntPair(3, 5)
    assert pair.a == 3 and pair.b == 5
    assert pair.get_first() == 3
    ```
- 错误排查
  - 未找到库：根据异常信息检查候选目录，或设置 `TVM_BOOK_LIB_PATH/TVM_BOOK_LIB_DIR`
  - VTA 未链接：相关 API 调用将抛出错误，需在构建时链接 VTA 运行时