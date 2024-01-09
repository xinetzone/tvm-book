import tvm
from tvm import relay
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.backend import te_compiler
from tqdm import tqdm

def update_lib(lib, source_dir="/media/pc/data/lxw/ai/tvm"):
    kwargs = {
        "options" : [
            "-O2", "-std=c++17", 
            "-I" + f"{source_dir}/src/runtime/contrib", 
            "-I" + f"{source_dir}/include",
            "-I" + f"{source_dir}/3rdparty/dlpack/include"
        ]
    }
    tmp_path = tvm.contrib.utils.tempdir()
    lib_name = "lib.so"
    lib_path = tmp_path.relpath(lib_name)
    lib.export_library(lib_path, fcompile=False, **kwargs)
    lib = tvm.runtime.load_module(lib_path)
    return lib

def run_llvm_graph(run_mod, params, input_dict, source_dir="/media/pc/data/lxw/ai/tvm"):
    target = "llvm" #"c -runtime=c --system-lib"
    device = tvm.cpu()
    # input_dict = {"data": data_np}
    te_compiler.get().clear()
    with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
        lib = relay.build(run_mod, target=target, params=params)
    lib = update_lib(lib, source_dir=source_dir)
    exe = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.cpu()))
    exe.run(**input_dict)
    tvm_res = [
        exe.get_output(k)
        for k in tqdm(range(exe.get_num_outputs()))
    ]
    return tvm_res

def run_llvm_vm(mod, params, input_dict):
    from tvm.runtime.vm import VirtualMachine
    with tvm.transform.PassContext(opt_level=3):
        vm_exec = relay.vm.compile(mod, target="llvm", params=params)
    vm = VirtualMachine(vm_exec, tvm.cpu())
    vm.set_input("main", **input_dict)
    tvm_res = vm.run()
    return tvm_res
