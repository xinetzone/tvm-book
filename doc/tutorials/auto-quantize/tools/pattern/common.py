from tvm.relay.dataflow_pattern import is_constant, is_op, wildcard, is_tuple_get_item

def make_activate(r):
    r1 = r.optional(lambda x: is_op("nn.relu")(x))
    r2 = r.optional(lambda x: is_op("clip")(x)) # relu6
    r3 = r.optional(lambda x: is_op("nn.prelu")(x, wildcard())) # prelu
    r4 = r.optional(lambda x: is_op("sigmoid")(x)) # sigmoid
    return r1 | r2 | r3 | r4

# batch_norm = is_op("nn.batch_norm")(conv_node, is_constant(), is_constant(), is_constant(), is_constant())
# batch_norm = is_tuple_get_item(batch_norm, 0)
