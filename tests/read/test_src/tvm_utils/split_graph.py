import tvm
from tvm import relay
from tvm.relay import transform, build_module
from tvm.relay.testing import run_opt_pass
# from tvm.contrib import graph_executor
# from tvm._ffi import get_global_func
# from tvm.contrib import cc as _cc


def graph_split(expr, split_conf, params=None):
    """Splitting the graph into a list of subgraphs
    
    e.g.：split_conf = [{"op_name": "nn.relu", "op_index": 0}]
    """

    def get_dep_var(sub_var_dep):
        return [var for var in sub_var_dep[len(sub_var_dep) - 1]["ref_nodes"]]

    def parse_dependency(value, snode_dep, new_input_idx):
        new_args = []
        need_update = False
        for var in value.args:
            is_free_var = False
            for dep in snode_dep[:-1]:
                if var in dep["nodes"]:
                    # Mark the previous subgraph node as a dependency.
                    dep["nodes"][var] += 1
                    dep["ref_nodes"][var] = dep["nodes"][var]
                    # The var of this call is a free_var
                    is_free_var = True
            # if the var of this call is a free_var, recreate it and give it a fixed input name.
            if is_free_var:
                need_update = True
                new_args.append(relay.var(f"data_n_{new_input_idx}", var.checked_type))
                new_input_idx += 1
            else:
                new_args.append(var)
        # if the 'tvm.relay.expr.Call' has a free_var, recreate it with new name as 'data_n_*'.
        if need_update:
            value = tvm.relay.expr.Call(
                value.op, new_args, value.attrs, value.type_args, value.span
            )
        return value, snode_dep, new_input_idx

    def merge_constant_expr(constant_expr, expr):
        # merge constant express with a express
        if not isinstance(constant_expr.body, tvm.relay.expr.Let):
            return tvm.relay.expr.Let(constant_expr.var, constant_expr.value, expr)

        return tvm.relay.expr.Let(
            constant_expr.var, constant_expr.value, merge_constant_expr(constant_expr.body, expr)
        )

    def _recursion(anf, pipeline_mods, split_conf, constant_expr):
        """列举计算图中的所有算子，然后将计算图分成一组子图。"""
        nonlocal operator_index_map
        nonlocal new_input_idx
        nonlocal snode_dep
        cur_node_dep = snode_dep[len(snode_dep) - 1]
        if isinstance(anf, tvm.relay.Function):
            return tvm.relay.Function(
                anf.params,
                _recursion(anf.body, pipeline_mods, split_conf, constant_expr),
                anf.ret_type,
                anf.type_params,
                anf.attrs,
            )
        elif isinstance(anf, tvm.relay.expr.Let):
            value = anf.value
            # 记录常量表达式，以确保所有子图都能找到正确的常量。
            if isinstance(value, tvm.relay.expr.Constant):
                if not constant_expr:
                    constant_expr = tvm.relay.expr.Let(anf.var, value, anf.var)
                else:
                    constant_expr = tvm.relay.expr.Let(anf.var, value, constant_expr)
            if isinstance(value, tvm.relay.expr.Call):
                new_args = []
                # 构建当前变量列表
                cur_node_dep["nodes"][anf.var] = 0
                # 获得节点的依赖信息。
                value, snode_dep, new_input_idx = parse_dependency(value, snode_dep, new_input_idx)
                if isinstance(value.op, tvm.ir.Op):
                    if value.op.name in operator_index_map:
                        operator_index_map[value.op.name] += 1
                    else:
                        operator_index_map[value.op.name] = 0
                    split_operator_name = split_conf[0]["op_name"] if split_conf else ""
                    split_operator_index = split_conf[0]["op_index"] if split_conf else ""
                    # 如果网络中的算子名称和重复计数与“分割配置”的值匹配，则应该在这里执行图分割。
                    if (
                        split_conf
                        and split_operator_name in operator_index_map
                        and operator_index_map[split_operator_name] >= split_operator_index
                    ):
                        # 执行图分割
                        split_conf.pop(0)
                        snode_dep.append({"nodes": {}, "ref_nodes": {}})
                        ann = _recursion(
                            anf.body,
                            pipeline_mods,
                            split_conf,
                            constant_expr,
                        )
                        snode_dep.pop()
                        dep_vars = get_dep_var(snode_dep)
                        # 当前子图的节点是另一个子图的依赖节点时，需要将它们设置为当前子图的输出。
                        body = relay.Tuple(dep_vars) if len(dep_vars) > 1 else anf.var
                        # 当当前子图的算子使用先前子图的常量作为 ``relay.expr.call`` 的参数时，如果该常量不在当前子图中，则可能会成为自由变量。为了避免这个问题，可以将先前的常量与当前子图合并。
                        if constant_expr:
                            ann = merge_constant_expr(constant_expr, ann)
                        ann = run_opt_pass(ann, transform.ToGraphNormalForm())
                        mod = tvm.IRModule.from_expr(ann)
                        pipeline_mods.insert(0, mod)
                        # 返回当前子图的最后一个节点。
                        return tvm.relay.expr.Let(anf.var, value, body)
            return tvm.relay.expr.Let(
                anf.var,
                value,
                _recursion(anf.body, pipeline_mods, split_conf, constant_expr),
            )
        else:
            return anf

    snode_dep = [{"nodes": {}, "ref_nodes": {}}]
    pipeline_mods = []
    operator_index_map = {}
    # Used to tracking new input which caused by graph splitting.
    new_input_idx = 0
    constant_expr = None
    subgraph_split_conf = split_conf.copy()
    # Binding the parameters.
    if params:
        expr = build_module.bind_params_by_name(expr, params)
    anf = run_opt_pass(expr, transform.ToANormalForm())
    anf = run_opt_pass(anf, transform.InferType())
    ann = _recursion(
        anf,
        pipeline_mods,
        subgraph_split_conf,
        constant_expr,
    )
    ann = run_opt_pass(ann.body, transform.ToGraphNormalForm())
    mod = tvm.IRModule.from_expr(ann)
    pipeline_mods.insert(0, mod)
    return pipeline_mods

