from tvm import te
import tvm

@tvm.target.generic_func
def schedule_special_op(attrs, outs, target):
    with target:
        outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
        output = outs[0]
        sch = te.create_schedule(output.op)
        return sch

@tvm.target.generic_func
def generic(data):
    # default generic function
    return data
