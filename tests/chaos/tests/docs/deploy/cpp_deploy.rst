..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

使用 C++ API 部署 TVM Module
===============================

`apps/howto_deploy <https://github.com/apache/tvm/tree/main/apps/howto_deploy>`_ 中提供了如何部署 TVM 模块的示例，可以使用以下命令运行该示例：


.. code:: bash

    cd apps/howto_deploy
    ./run_example.sh


获取 TVM 运行时库
-----------------------

只需要链接到目标平台中的 TVM 运行时。TVM 提供了最小运行时，根据使用的模块数量，运行时的消耗大约在 300K 到 600K 之间。
在大多数情况下，可以使用随 ``build`` 而来的 ``libtvm_runtime.so``。

如果您发现构建 ``libtvm_runtime`` 很困难，
请检出 `tvm_runtime_pack.cc <https://github.com/apache/tvm/tree/main/apps/howto_deploy/tvm_runtime_pack.cc>`_。
这是在一个文件中提供 TVM 运行时的示例。
您可以使用构建系统编译此文件，并将其包含到项目中。

你也可以签出 `apps <https://github.com/apache/tvm/tree/main/apps/>`_，例如在 iOS, Android 和其他平台上用 TVM 构建的应用程序。

动态库 vs. 系统模块
---------------------------------

TVM 提供了两种使用编译库的方法。
您可以签出 `prepare_test_libs.py <https://github.com/apache/tvm/tree/main/apps/howto_deploy/prepare_test_libs.py>`_ 关于如何生成库
和 `cpp_deploy.cc <https://github.com/apache/tvm/tree/main/apps/howto_deploy/cpp_deploy.cc>`_ 关于如何使用它们。

- 将库存储为共享库，并将库动态加载到项目中。
- 以系统模块（module）模式将编译后的库捆绑到项目中。

动态加载更加灵活，可以动态加载新模块。系统模块是更 ``static`` 的方法。可以在禁止动态库加载的地方使用系统模块。
