"""Example extension package of TVM."""
from __future__ import absolute_import
import os
import ctypes

# Import TVM first to get library symbols
import tvm
from tvm import te


def load_lib():
    """Load library, the functions will be registered into TVM"""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    # load in as global so the global extern symbol is visible to other dll.
    lib = ctypes.CDLL(os.path.join(curr_path, "../../outputs/libs/libtvm_ext.so"), ctypes.RTLD_GLOBAL)
    return lib


_LIB = load_lib()

# Expose two functions into python
bind_add = tvm.get_global_func("tvm_ext.bind_add")
sym_add = tvm.get_global_func("tvm_ext.sym_add")
ivec_create = tvm.get_global_func("tvm_ext.ivec_create")
ivec_get = tvm.get_global_func("tvm_ext.ivec_get")


@tvm.register_object("tvm_ext.IntVector")
class IntVec(tvm.Object):
    """Example for using extension class in c++"""

    @property
    def _tvm_handle(self):
        return self.handle.value

    def __getitem__(self, idx):
        return ivec_get(self, idx)


nd_create = tvm.get_global_func("tvm_ext.nd_create")
nd_add_two = tvm.get_global_func("tvm_ext.nd_add_two")
nd_get_additional_info = tvm.get_global_func("tvm_ext.nd_get_additional_info")


@tvm.register_object("tvm_ext.NDSubClass")
class NDSubClass(tvm.nd.NDArrayBase):
    """Example for subclassing TVM's NDArray infrastructure.

    By inheriting TVM's NDArray, external libraries could
    leverage TVM's FFI without any modification.
    """

    @staticmethod
    def create(additional_info):
        return nd_create(additional_info)

    @property
    def additional_info(self):
        return nd_get_additional_info(self)

    def __add__(self, other):
        return nd_add_two(self, other)
