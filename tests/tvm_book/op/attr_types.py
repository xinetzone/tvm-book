from enum import IntEnum

class OpPatternKind(IntEnum):
    """算子模式类型枚举

    定义了 TVM 中不同类型的算子模式，用于优化器决策、算子融合和代码生成等。
    每个枚举值对应一种特定的算子模式类型。
    """
    # 元素级算子，逐元素进行计算的算子，如 add、mul 等
    kElemWise = 0

    # 广播算子：可以将输入的维度广播到输出维度的算子
    # 例如：out[i, ax1, j, ax2] = input[i, j]
    # 注意：轴必须按顺序排列，因此转置算子不是广播算子
    kBroadcast = 1

    # 单射算子：输出轴可以单射映射到单个输入轴的算子
    # 所有单射算子可以安全地与单射算子和归约算子融合
    kInjective = 2

    kCommReduce = 3 # 通信归约算子
    kOutEWiseFusable = 4 # 复杂算子，但仍可融合元素级算子，但不能链接另一个复杂算子
    kTuple = 7 # 可以融合到后续的单射算子中，但会被特殊处理
    kOpaque = 8 # 不可融合的不透明算子
