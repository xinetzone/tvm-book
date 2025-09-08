from tvm.relax import expr as _expr

def _forward_op(ref_call, args):
    """forward the operator of ref_call with provided arguments"""
    return _expr.Call(ref_call.op, args, ref_call.attrs, ref_call.sinfo_args, ref_call.span)


