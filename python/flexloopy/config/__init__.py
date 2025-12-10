"""FlexLoopy 配置包

本包用于集中管理与项目相关的环境配置，当前包含：

- `env`：提供 ``set_tvm`` 与 ``set_caffeproto`` 两个函数，用于设置 TVM/VTA 以及 CaffeProto 的 Python 路径与相关环境变量。

使用示例：

    >>> from flexloopy.config.env import set_tvm, set_caffeproto
    >>> set_tvm("/path/to/tvm")
    >>> set_caffeproto("/path/to/CaffeProto")

说明：

    - 包级文档仅做目录与入口说明，不改变任何运行逻辑。
"""
