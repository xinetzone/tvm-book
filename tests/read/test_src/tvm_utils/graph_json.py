"""解析 Graph JSON
"""
from typing import Any
from dataclasses import dataclass, field


@dataclass
class GraphNodeRef:
    ident: int  # 节点引用索引
    index: int = 0  # 暂不知作用
    version: int = 0  # 暂不知作用


from dataclasses import field


@dataclass
class GraphAttrs:
    """`
    Args:
        dltype: 每个节点的数据类型按顺序排列。
        device_index: 按顺序为每个节点分配设备。
        storage_id: 存储布局中每个节点的内存 slot id。
        shape: 每个节点的 k 阶形状。
        storage_id: 存储布局中每个节点的内存 slot id。
                    将参数名称映射到一对 ({storage_id: tvm.runtime.NDArray})。在运行时，可以使用 storage_id 查找参数。
    """
    dltype: list
    storage_id: list
    shape: list
    device_index: list[int] = field(default_factory=list)


@dataclass
class GraphNodeAttrs:
    """
    Args:
        flatten_data: 是否需要在执行前将数据扁平化（flattened）
        func_name: 融合函数名，对应于 Relay 编译过程生成的库中的符号。
        num_inputs: 此节点的 inputs 个数
        num_outputs: 此节点产生的 outputs 个数
    """
    func_name: str
    num_inputs: str
    num_outputs: str
    flatten_data: str = "0"
    hash: str|None = None
    


@dataclass
class GraphNode:
    """
    Args:
        op: 运算类型，`null` 意味着它是占位符/变量/输入节点，`tvm_op` 意味着这个节点可以被执行
        name: 节点名字
        inputs: 运算的 inputs 位置，inputs 是包含 `(nodeid, index, version)` 的元组列表。(可选)
    """
    op: str
    name: str
    inputs: list[int] = field(default_factory=list)
    attrs: Any = None

    # def __post_init__(self):
    #     if self.op == "null":
    #         delattr(self, "attrs")


@dataclass
class GraphJson:
    """
    Args:
        arg_nodes:参数节点的索引列表，它是计算图的占位符/变量/输入节点或 constant/param。
        heads: 输出节点的索引列表。
        node_row_ptr: 存储 forward 路径的历史，所以推断任务中可以跳过某些算子来构建子图。
        attrs: 可以包含版本号或类似的有用信息。
        nodes: 节点是占位符或可计算节点。
    """
    arg_nodes: list[int]
    heads: list[GraphNodeRef]
    node_row_ptr: list[int]
    attrs: GraphAttrs
    nodes: list[GraphNode]

    def __post_init__(self):
        self.heads = [GraphNodeRef(*head) for head in self.heads]
        self.attrs = GraphAttrs(**self.attrs)
        self.nodes = [GraphNode(**node) for node in self.nodes]


