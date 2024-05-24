import sys
from pathlib import Path
ROOT = Path(".").resolve().parents[4]
sys.path.extend([f"{ROOT}/tests", f"{ROOT}/src"])
# # from tools.tag_span import _create_span, _set_span, _verify_structural_equal_with_span
from tools.torch_utils import verify_model
import tvm
from tvm.contrib.relay_viz import RelayVisualizer
from tvm.contrib.relay_viz.dot import DotPlotter
from graphviz import Digraph
from IPython.display import display_svg


class Visualizer(RelayVisualizer):
    def graph(self, graph_name):
        return self._plotter._name_to_graph[graph_name]

    def display(self, graph_name):
        graph = self.graph(graph_name)
        return graph.digraph

    def display_all(self, format="svg",
                    filename=None,
                    directory="images"):
        root_graph = Digraph(format=format,
                             filename=filename,
                             directory=directory)
        for graph in self._plotter._name_to_graph.values():
            root_graph.subgraph(graph.digraph)
        return root_graph
    
# VizNode is passed to the callback.
# We want to color NCHW conv2d nodes. Also give Var a different shape.
def get_node_attr(node):
    if "nn.conv2d" in node.type_name and "NCHW" in node.detail:
        return {
            "fillcolor": "green",
            "style": "filled",
            "shape": "box",
        }
    if "Var" in node.type_name:
        return {"shape": "ellipse"}
    return {"shape": "box"}

def viz_expr(expr, func_name="main"):
    graph_attr = {"color": "red"}
    node_attr = {"color": "blue"}
    edge_attr = {"color": "black"}
    
    # 添加颜色
    dot_plotter = DotPlotter(
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr,
        get_node_attr=get_node_attr
    )
    mod = tvm.IRModule.from_expr(expr)
    viz = Visualizer(mod, plotter=dot_plotter)
    graph = viz.display(func_name)
    display_svg(graph)
