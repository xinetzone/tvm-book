"""tvm_ext.testing FFI"""
import tvm._ffi

tvm._ffi._init_api("tvm_ext", __name__)
