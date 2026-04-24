# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
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
