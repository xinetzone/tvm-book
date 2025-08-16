import tvm

def check_unsupported_ops(predict_layer, supported_op_names):
    """Check unsupported Caffe ops in our converter."""
    unsupported_ops_set = set()

    include_layer = dict()
    for pl in predict_layer:
        if pl.type not in include_layer:
            include_layer[pl.type] = 1
        else:
            include_layer[pl.type] = include_layer[pl.type] + 1

    for pl in predict_layer:
        op_name = pl.type
        if op_name not in supported_op_names:
            unsupported_ops_set.add(op_name)

    if unsupported_ops_set:
        msg = "The following operators are not supported in frontend " "Caffe: {}"
        ops = str(list(unsupported_ops_set)).strip("[,]")
        raise tvm.error.OpNotImplemented(msg.format(ops))
