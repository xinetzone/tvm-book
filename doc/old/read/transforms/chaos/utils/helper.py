import tvm
from tvm import relay
from tvm.relay import ExprFunctor
from tvm.relay import Function, Call


def get_var_func():
    shape = (5, 10)
    tp = relay.TensorType(shape, "float32")
    x = relay.var("x", tp)
    gv = relay.GlobalVar("myAbs")
    func = relay.Function([x], relay.abs(x))
    return gv, func


def extract_var_func(mod, name):
    var = mod.get_global_var(name)
    func = mod[var]
    return var, func


def update_func(func):
    # Double the value of Constants and vars.
    class DoubleValues(ExprFunctor):
        def __init__(self):
            ExprFunctor.__init__(self)

        def visit_constant(self, const):
            return relay.add(const, const)

        def visit_var(self, var):
            return relay.add(var, var)

        def visit_call(self, call):
            new_op = self.visit(call.op)
            new_args = [self.visit(arg) for arg in call.args]
            return Call(new_op, new_args, call.attrs)

        def visit_global_var(self, gvar):
            return gvar

        def visit_op(self, op):
            return op

        def visit_function(self, fn):
            new_body = self.visit(fn.body)
            return Function(list(fn.params), new_body, fn.ret_type, fn.type_params, fn.attrs)

    double_value = DoubleValues()
    return double_value.visit(func)

class OptTester:
    """A helper class for testing the pass manager."""

    def __init__(self, mod):
        if not isinstance(mod, tvm.IRModule):
            raise TypeError("mod is expected to be the type of " "tvm.IRModule")
        self.mod = mod

    def analysis(self):
        """Perform analysis for the current module."""
        pass

    @staticmethod
    def transform(node, ctx=None):
        """Perform optimization on node."""
        if isinstance(node, tvm.IRModule):
            # Add a function to the module and return an updated module.
            gv, func = get_var_func()
            mod = tvm.IRModule({gv: func})
            mod.update(node)
            return mod
        if isinstance(node, relay.Function):
            return update_func(node)
        raise TypeError("Found not supported node type.")

