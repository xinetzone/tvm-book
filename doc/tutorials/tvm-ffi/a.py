import numpy as np
import tvm_ffi

# Demonstrate DLPack conversion between NumPy and TVM FFI
np_data = np.array([1, 2, 3, 4], dtype=np.float32)
tvm_array = tvm_ffi.from_dlpack(np_data)
# Convert back to NumPy
np_result = np.from_dlpack(tvm_array)