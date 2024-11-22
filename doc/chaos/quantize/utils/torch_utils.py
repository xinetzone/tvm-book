import torch

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.empty_cache()


def list_ops(expr):
    """list_ops"""

    class OpLister(tvm.relay.ExprVisitor):
        """OpLister inherits from ExprVisitor"""

        def visit_op(self, op):
            if op not in self.node_set:
                self.node_list.append(op)
            return super().visit_op(op)

        def list_nodes(self, expr):
            self.node_set = {}
            self.node_list = []
            self.visit(expr)
            return self.node_list

    return OpLister().list_nodes(expr)