import tvm
# 参考：https://tvm.apache.org/docs/how_to/work_with_relay/using_pipeline_executor.html?highlight=graph_split
from tvm_book.tvm_utils.split_graph import graph_split
split_conf = [{"op_name": "nn.max_pool2d", "op_index": 0}]
pipeline_mods = graph_split(mod["main"], split_conf, params)
run_mod = pipeline_mods[0]