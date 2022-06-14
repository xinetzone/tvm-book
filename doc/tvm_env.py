from platform import python_version

from tvm_book.contrib.tvm.set_env import set_tvm
_python_version = python_version()

if _python_version == '3.8.10':
    TVM_ROOT = "/media/pc/data/4tb/lxw/tvm310"
else:
    TVM_ROOT = "/media/pc/data/4tb/lxw/books/tvm"
    # TVM_ROOT = "/media/pc/data/4tb/lxw/books/tvm"

set_tvm(TVM_ROOT)


if __name__ == "__main__":
    import tvm
    print(f'Python: {_python_version}')
    print(f'TVM: {tvm.__version__}')
    # tvm, vta = tvmx.import_tvm(TVM_ROOT)
    # print("TVM 根目录：", TVM_ROOT)
