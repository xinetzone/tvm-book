# from tvm import relay
# from tvm.ir import IRModule
# from IPython.display import display_svg
from tvm.contrib.relay_viz.dot import (
    DotPlotter,
    DotVizParser
)
from tvm_book.tvm.viz_relay import Visualizer


def graphviz_relay(ir_mod,
                   graph_name="main",
                   graph_attr={"color": "red"},
                   node_attr={"color": "blue"},
                   edge_attr={"color": "black"}):
    # 添加颜色
    dot_plotter = DotPlotter(
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr)
    viz = Visualizer(
        ir_mod,
        plotter=dot_plotter,
        parser=DotVizParser())
    return viz.display(graph_name)
    # display_svg(graph)
