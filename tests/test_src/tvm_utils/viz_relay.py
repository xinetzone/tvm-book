from graphviz import Digraph
from tvm.contrib.relay_viz import RelayVisualizer


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


if __name__ == "__main__":
    from tvm import relay
    from tvm.ir import IRModule
    from IPython.display import display_svg
    from tvm.contrib.relay_viz.dot import (
        DotPlotter,
        DotVizParser
    )
    from tvm_book.tvm.viz_relay import Visualizer


    x = relay.var("x")
    sb = relay.ScopeBuilder()
    v1 = sb.let("v1", relay.log(x))
    v2 = sb.let("v2", v1 + v1)
    sb.ret(v2)
    f = relay.Function([x], sb.get())

    mod = IRModule()
    mod["main"] = f

    graph_attr = {"color": "red"}
    node_attr = {"color": "blue"}
    edge_attr = {"color": "black"}

    # 添加颜色
    dot_plotter = DotPlotter(
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr)
    viz = Visualizer(
        mod,
        plotter=dot_plotter,
        parser=DotVizParser())
    graph = viz.display("main")
    display_svg(graph)