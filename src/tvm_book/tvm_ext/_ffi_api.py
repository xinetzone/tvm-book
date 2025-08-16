"""FFI APIs for TVM extension."""
import tvm._ffi

tvm._ffi._init_api("tvm_ext", __name__)
