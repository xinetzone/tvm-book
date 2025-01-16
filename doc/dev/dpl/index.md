# Relax 数据流模式语言

Relax 引入了一种名为 数据流模式语言（Dataflow Pattern Language，简称 DPL）的模式匹配语言，用于描述和匹配计算图中的特定模式，从而支持更灵活的图优化。

```{admonition} 核心思想
DPL 的核心思想是通过声明式的方式描述计算图中的子图模式。这种语言允许开发者定义计算图中的特定结构（如算子组合、数据流模式等），并基于这些模式进行优化或变换。

- 模式匹配：通过定义模式，可以在计算图中查找符合特定结构的子图。
- 图优化：匹配到特定模式后，可以对其进行替换、重写或优化。
```

```{toctree}
:hidden:

match
rewrite
PatternContext
rewrite-without-trivial-binding
same-shape
iterative-rewrite
StructInfo
backtrack
```
