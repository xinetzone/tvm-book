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

.. _pass-infra:

Pass Infrastructure
===================

Relay 和 TVM IR 都包含一系列优化 passes，用于改善模型的性能指标，如 mean inference、memory footprint 或特定设备的 power consumption。
有一套标准的优化和机器学习特有的优化，包括常量折叠（constant folding）、死代码消除（ead code
elimination）、算子布局更改（operator layout alteration）、算子融合（operator fusion）、buffer 处理和 loop transformation 等。
使用遍历（traversal）期间和/或遍历之前收集的分析结果，将每一个 passes 构造为 ir-to-ir 变换。

然而，随着 TVM 的迅速发展，对管理这些 passes 的更系统和有效的方法的需求变得越来越明显。
另外，管理跨 TVM 栈不同层（如 Relay 和 tir）的传递的通用框架为开发人员快速原型化和将实现的传递插入到系统中铺平了道路。

本文档描述了这样的基础设施（infra）的设计，它利用了生产编译器（production compiler）管理优化过程的方式，以及现代深度学习框架用于构建层的风格。

例如，许多现有的生产编译器，如 GCC 和 LLVM，都采用了传递管理器来有效地管理传递的执行。
最初管理传递是简单的，因为传递的次数很少，但成熟的编译器将包含数百次单独的传递。
外部用户通常希望能够正确地调度自定义传递，而不需要修改手工制作的传递 order。

同样，现代的深度学习框架，如 Pytorch 和 MXNet Gluon，也有分别通过 `Sequential`_ 和 `Block`_ 实现 pass-style 层构造方案的趋势。
有了这样的结构，这些现代框架能够方便地将模块/层添加到它们的容器中，并轻松地构建神经网络。

Relay pass infra 的设计很大程度上受到 LLVM 中使用的分层 pass 管理器和流行的深度学习框架中使用的 block-style 容器的启发。pass infra 的主要目标包括

#) 支持更好的优化编程编排（orchestration）。这允许用户灵活地定制和构建自己的优化管道。

#) 提供一种用户友好的方式来调试优化 passes。

#) 减轻开发人员手工修改和分别解决传递之间的依赖关系。

#) 简化开发人员实现新的 passes 的过程。例如，允许用户在 Python 中实现 pass，并让 pass infra 操作它的执行。

设计
----------

专注于用户扩展的易用性，使用户能够在不损失向后兼容性的情况下快速添加新的 passes。
该设计包括后端和前端。前者实现了 passes 底层的主要逻辑。后者提供了简单的 API 供用户交互，即允许用户快速创建自己的优化管道。

C++ 后端
~~~~~~~~~~~

提供 ``PassInfo`` 对象来包含 pass 所需的基本信息。
``name`` 是 pass 名称，``opt_level`` 表示在哪个优化级别将启用传递，``required`` 表示执行某个传递所需的传递（有关详细信息，请参阅 `include/tvm/ir/transform.h`_ ）。
例如，在 pass 的注册过程中（将在后面讨论），pass 开发人员可以指定 pass 的名称，它将执行的优化级别，和/或所需的 pass。
``opt_level`` 可用于帮助 pass infra 识别在用户提供的优化级别下运行时是否需要执行某个 pass。
``required`` 字段可以被 pass infra 用来解析传递依赖关系。

.. code:: c++

    class PassInfoNode : public Object {
      String name;
      int opt_level;
      Array<String> required;
    };

PassContext
^^^^^^^^^^^

``PassContext`` 为优化传递携带有用的信息。例如，它包含错误报告系统，这样优化作者就可以对优化失败的原因进行诊断。
``PassContext`` 还被设计用来取代旧的 ``BuildConfig``，后者用于帮助用户配置编译选项，包括优化级别和所需/禁用 pass 等。
例如，可能有一个配置，使用 ``PassContext`` 提供的 ``disabled_pass=xx`` 在 ``opt_level=3`` 执行所有的传递，并禁用一些传递。
现在，可以 glob ``opt_level=3`` 的所有传递，并排除那些在禁用的传递列表中。``PassContext`` 还提供了一种方法来检测（instrument）所有的传递。
参考 :ref:`pass_instrument_cpp_backend` 部分。

这个类是为用户设计的，用户可以方便地编写 Python ``with`` 语法，在特定的配置下执行优化。
此外，用户可以通过 ``PassContext::Current()`` 以线程安全的方式获得在某个程序范围内可用的上下文，
因为线程本地存储 ``PassContextThreadLocalStore`` 用于保存创建的传递上下文对象。
后面将提供一些示例，展示如何使用 C++ 和 Python API 使用传递上下文创建编译管道。

.. code:: c++

    class PassContextNode : public Object {
     public:
      int opt_level{2};
      tvm::Array<tvm::Expr> required_pass;
      tvm::Array<tvm::Expr> disabled_pass;
      mutable Optional<DiagnosticContext> diag_ctx;
      Map<String, ObjectRef> config;
      Array<instrument::PassInstrument> instruments;
    };

    class PassContext : public NodeRef {
     public:
      TVM_DLL static PassContext Create();
      TVM_DLL static PassContext Current();
      TVM_DLL void InstrumentEnterPassContext();
      TVM_DLL void InstrumentExitPassContext();
      TVM_DLL bool InstrumentBeforePass(const IRModule& mod, const PassInfo& info) const;
      TVM_DLL void InstrumentAfterPass(const IRModule& mod, const PassInfo& info) const;
      /* Other fields are omitted. */

     private:
      // The entry of a pass context scope.
      TVM_DLL void EnterWithScope();
      // The exit of a pass context scope.
      TVM_DLL void ExitWithScope();

      // Classes to get the Python `with` like syntax.
      friend class tvm::With<PassContext>;
    };

    struct PassContextThreadLocalEntry {
      /*! \brief The default pass context. */
      PassContext default_context;
      /*! \brief The current pass context. */
      std::stack<PassContext> context_stack;
      PassContextThreadLocalEntry() {
        default_context = PassContext(make_node<PassContextNode>());
      }
    };

    /*! \brief The thread-local store to hold the pass context. */
    typedef dmlc::ThreadLocalStore<PassContextThreadLocalEntry>
         PassContextThreadLocalStore;

Pass 构建
^^^^^^^^^^^^^^^

pass infra 是分层设计的，可以在不同粒度的 Relay/tir 程序中工作。
引入纯虚类 ``PassNode`` 作为不同优化 passes 的基础。该类包含几个虚方法，这些方法必须由模块、函数或 pass 序列级别的子类实现。

.. code:: c++

    class PassNode : Object {
      virtual PassInfo Info() const = 0;
      virtual Module operator()(const IRModule& mod
                                const PassContext& pass_ctx) const = 0;
    };

functor 展示了 pass 必须如何实现，也就是说，它总是在 :py:class:`IRModule` 的特定上下文中工作。
所有的 pass 都是以 ``Module`` 到 ``Module`` 的方式设计的。
因此，由 pass infra 管理的优化将始终更新整个模块。

已经创建了几个子类来实现不同类型的优化传递，例如，函数级传递、模块级传递和序列级传递。
每个子类本身可以充当 pass 管理器。例如，可以收集所需的传递并执行它们，或者基于给定的元数据构建依赖关系图。
它们的完整定义可以在 `src/relay/ir/transform.cc`_ 和 `src/ir/transform.cc`_ 中找到。

模块级 Passes
^^^^^^^^^^^^^^^^^^^

模块级 passes 主要用于全局和过程间优化（inter-procedural optimizations，简称 IPO），这与 LLVM 中使用的模块 passes 类似。
Relay 中一些需要模块 global picture 的典型过程，如 A-normal form conversion、lambda lifting 等，都属于这个集合。
在这个级别上，用户甚至可以在模块中添加和/或删除函数。注意，所有的 passes

.. code:: c++

    class ModulePassNode : PassNode {
      PassInfo pass_info;
      runtime::TypedPackedFunc<Module(Module, PassContext)> pass_func;
      Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
      // Other members/methods are omitted
    };


``pass_info`` 维护模块级 pass 所需的信息。 ``pass_func`` 描述了真正的优化。例如，可能需要在模块上执行死代码消除。
可以在 ``pass_func`` 中实现算法，并让它在模块上运行。然后，它将删除死代码，包括模块中未使用的函数。
请注意，该字段被设计为 packed function，它支持在 C++ 和 Python 中实现优化。

函数级 Passes
^^^^^^^^^^^^^^^^^^^^^

函数级 Pass 用于为给定的 Relay/tir 模块实现各种内部函数级优化。
它每次从模块的函数列表中获取一个函数用于优化，并生成重写的 Relay ``Function`` 或 tir ``PrimFunc``。
大多数的 Pass 都可以归为这一类，如 Relay 中常见的子表达式消除（subexpression elimination）和推理简化（inference simplification），
以及 tir 中的向量化和扁平化存储（flattening storage）等。

注意，这个级别的 pass 的作用域是 Relay 函数或 tir 原语函数。因此，不能通过这些 pass 添加或删除函数，因为它们不知道全局信息。

.. code:: c++

    class FunctionPassNode : PassNode {
      PassInfo pass_info;
      runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func;
      Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
      bool SkipFunction(const Function& func) const;
      // Other members/methods are omitted...
    };

``pass_info`` 与刚才在模块级 pass 中描述的内容相同。
``pass_func`` 接受函数来进行优化，它还需要模块，因为可能会使用它来报告错误。
函数可以用 "SkipOptimization" 进行注解，以便在优化过程中忽略它。

序列级 Passes
^^^^^^^^^^^^^^^^^

``SequentialPass`` 类似于 ``nn.Sequential``，包含一系列执行过程的序列。

.. code:: c++

    class SequentialPassNode : PassNode {
      PassInfo pass_info;
      // Passes need to be executed.
      Array<Pass> passes;
      bool PassEnabled(const PassInfo& info) const;
      Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
    };

目前在 Relay 中只有少数的 passes 被放在这个组中。
例如， ``FoldScaleAxis`` 需要内部分派 ``ForwardFoldScaleAxis`` 和 ``BackwardFoldScaleAxis``。
此外，建议首先完成 ``BackwardFoldScaleAxis``。因此，该 pass 是 ``SequentialPass`` 的理想候选者。

下面的代码展示了如何调用序列 passes 中的单个 pass。从本质上讲，使用添加到 passes 列表的顺序，在 psss 序列中顺序地执行每个 pass。

.. code:: c++

    Module SequentialNode::operator()(const Module& module,
                                      const PassContext& pass_ctx) const {
      Module mod = module;
      for (const Pass& pass : passes) {
        ICHECK(pass.defined()) << "Found undefined pass for optimization.";
        const PassInfo& pass_info = pass->Info();
        if (!PassEnabled(pass_info))  continue;
        for (const auto& it : pass_info->required) {
          const auto* name = it.as<tvm::ir::StringImm>();
          ICHECK(name);
          mod = GetPass(name->value)(mod, pass_ctx);
        }
        mod = pass(mod, pass_ctx);
      }
      return mod;
    }

在调用 pass 时，首先检查这个 pass 是否启用。
首先检查 pass 是否被用户显式禁用，然后检查它是否被用户指定为必需的 pass。
如果仍然不确定是否启用这个 pass，那么将检查它的 ``opt_level``。
只有当它的优化级别不低于在 pass 上下文中配置的优化级别时，该 pass 才会启用并执行。

要执行 pass，首先需要使用 pass 名在 TVM 打包的函数注册表中检索注册的 pass。
这是可能的，因为每一个 pass 都是用 API endpoint 注册的，将在后面展示。

.. code:: c++

    Pass GetPass(const std::string& pass_name) {
      using tvm::runtime::Registry;
      std::string fpass_name = "relay._transform." + pass_name;
      const auto* f = Registry::Get(fpass_name);
      ICHECK(f != nullptr) << "Cannot find " << fpass_name
                          << "to create the pass " << pass_name;
      return (*f)();
    }

提供了一些辅助函数来创建上述每种类型的 pass。这些辅助程序还暴露在 Python 前端，以便用户使用 Python API 创建特定的 pass 对象。

.. code:: c++

    Pass CreateFunctionPass(
        const runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>& pass_func,
        int opt_level,
        String name,
        Array<String> required);

    Pass CreatePrimFuncPass(
        const runtime::TypedPackedFunc<PrimFunc(PrimFunc, IRModule, PassContext)>& pass_func,
        int opt_level,
        String name,
        Array<String> required);

    Pass CreateModulePass(
        const runtime::TypedPackedFunc<IRModule(IRModule, PassContext)>& pass_func,
        int opt_level,
        String name,
        Array<String> required);

    Pass Sequential(tvm::Array<Pass> passes, PassInfo pass_info);

Pass 注册
^^^^^^^^^^^^^^^^^

已经介绍了不同级别的 pass 的概念以及编译时使用的上下文。
看看用户注册 pass 有多容易，这将是一件有趣的事情。以 const 折叠为例。
这个 pass 已经实现了在 Relay 函数中折叠常量（参见 `src/relay/transforms/fold_constant.cc`_）。

提供了 API 来执行 ``Expr`` 到 ``Expr`` 的变换。

.. code:: c++

    Expr FoldConstant(const Expr& expr);

为了将此 pass 注册到 pass infra，首先需要决定在哪个级别执行此 pass。
由于 const 折叠发生在单个函数上，应该通过 ``CreateFunctionPass`` 直观地为它创建 ``FunctionPass``。
``pass_func`` 作为打包函数返回，它调用了 ``IRModule`` 中每个函数的 ``Expr`` 到 ``Expr`` 的 API。
``{}`` 表示此 pass 不需要任何先决条件。否则，pass 开发人员必须识别并列出它们。

同时，使用 ``relay._transform.FoldConstant`` 名称注册 pass API 端点。
因此，这个 pass 成为注册表中的条目，C++（例如上面的 ``GetPass``）和 Python 在需要时都可以访问它。

.. code:: c++

    namespace transform {

    Pass FoldConstant() {
      runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
        [=](Function f, IRModule m, PassContext pc) {
          return Downcast<Function>(FoldConstant(f));
      };
      return CreateFunctionPass(pass_func, 2, "FoldConstant", {});
    }

    TVM_REGISTER_GLOBAL("relay._transform.FoldConstant")
    .set_body_typed(FoldConstant);

    }  // namespace transform

为了允许其他 C++ 模块应用此 pass，在 `include/tvm/relay/transform.h`_  中声明自由函数（free function）：

.. code:: c++

    TVM_DLL Pass FoldConstant();

.. _pass_instrument_cpp_backend:

Pass Instrument
^^^^^^^^^^^^^^^

Pass Instrument 是分析 Pass 自身的机制。
例如，可以使用 infrastructure 来获知 pass 需要多少时间和内存，或者 pass 如何变换 IR 模块。

引入了 ``PassContext`` 生命周期中的四个 instrument 点。

.. code:: c++

    TVM_DLL void InstrumentEnterPassContext();
    TVM_DLL void InstrumentExitPassContext();
    TVM_DLL bool InstrumentBeforePass(const IRModule& mod, const PassInfo& info) const;
    TVM_DLL void InstrumentAfterPass(const IRModule& mod, const PassInfo& info) const;

当进入 ``PassContext`` 实例的作用域时，立即调用 ``InstrumentEnterPassContext``。

当离开 ``PassContext`` 的作用域，或者在 pass 的执行过程中发生异常时，将调用 ``InstrumentExitPassContext``。
当在 :py:class:`tvm.transform.PassContext` 中被 ``override_instruments`` 重写 instruments 时，也会调用此方法。
请参阅 :ref:`pass_instrument_overriden`。

在执行之前调用 ``InstrumentBeforePass``。如果通过，则在执行后调用 ``InstrumentAfterPass``。行为编写如下：

.. code:: c++

      if (pass_ctx.InstrumentBeforePass(ir_module, pass_info)) {
        new_ir_module = run_pass(ir_module, pass_ctx);
        pass_ctx.InstrumentAfterPass(new_ir_module, pass_info);
        return new_ir_module;
      }

``PassInstrument`` 接口允许您在以上四个方法中运行任意代码。
可以将多个 ``PassInstrument`` 实例注册到单个 ``PassContext`` 中。
``PassInstrument`` 实例按照传递给 ``PassContext`` 的 ``instruments`` 参数的顺序被调用。

``PassInstrument`` 提供的接口有：

.. code:: c++

    namespace instrument {

    class PassInstrumentNode : public Object {
     public:
      String name;
      virtual void EnterPassContext() const = 0;
      virtual void ExitPassContext() const = 0;
      virtual bool ShouldRun(const IRModule& mod, const transform::PassInfo& info) const = 0;
      virtual void RunBeforePass(const IRModule& mod, const transform::PassInfo& info) const = 0;
      virtual void RunAfterPass(const IRModule& mod, const transform::PassInfo& info) const = 0;
      /* Other fields are omitted. */
    };

    class PassInstrument : public ObjectRef {
     public:
      TVM_DEFINE_OBJECT_REF_METHODS(PassInstrument, ObjectRef, PassInstrumentNode);
    };

    }  // namespace instrument

提供了 Python 前端来快速实现 ``PassInstrument``。参阅 :ref:`pass_instrument_py_frontend`。

在 ``PassContext`` 中，``PassInstrument`` 实例的调用序列如下：

::

    with PassContext(instruments=[pi]) # pi = a PassInstrument implementation.
        pi.EnterPassContext()

        if pi.ShouldRun(Pass1):
            pi.RunBeforePass()
            Pass1()
            pi.RunAfterPass()

        if pi.ShouldRun(Pass2):
            pi.RunBeforePass()
            Pass2()
            pi.RunAfterPass()

        pi.ExitPassContext()

下面简要介绍 ``PassInstrument`` 接口和 ``PassContext`` 方法之间的关系。阅读 (`src/ir/transform.cc`_) 了解更多细节。

- ``InstrumentEnterPassContext``

  * ``EnterPassContext()`` 是按照传递给 ``PassContext`` 的 ``instruments`` 顺序执行的。
  * 当异常触发时， ``PassContext`` 通过清除所有注册的 ``PassInstrument`` 实例来禁用 pass 检测（instrumentation）。
  * 然后 ``PassContext`` 对每个成功完成 ``EnterPassContext()`` 的 ``PassInstrument`` 实例执行 ``ExitPassContext()`` 方法
  * 例如，如果 ``PassInstrument`` A、B 和 C 被注册到 ``PassContext``，A 完成了 ``EnterPassContext()``，而 B 抛出异常，那么 C 永远不会被执行；执行 A 的 ``ExitPassContext()``。


- ``InstrumentExitPassContext``

  * 每个 ``PassInstrument`` 实例的 ``ExitPassContext()`` 将按照传递给 ``PassContext`` 的 ``instruments`` 顺序执行。
  * 当触发异常，则 ``instruments`` 被清除。
  * ``PassInstrument`` 在抛出异常之后注册的 ``PassInstrument`` 实例不执行  ``ExitPassContext``。

- ``InstrumentBeforePass``

  * ``ShouldRun`` is executed if the pass is not listed as a required pass.
  * ``RunBeforePass`` is executed in the order of ``instruments`` if the pass is not blocked by ``ShouldRun``.
  * Note that ``InstrumentBeforePass`` returns a boolean indicating whether or not the pass should be run.
  * When an exception occur, it is thrown immediately.
    We rely on Python Context Manager to exit ``PassContext`` safely
    (meaning ``ExitPassContext`` of each instruments will be run. For C++, please refer to `include/tvm/support/with.h`_.)

- ``InstrumentAfterPass``

  * ``RunAfterPass`` is executed in the order of ``instruments`` passed to the ``PassContext``.
  * When an exception occur, it is thrown immediately.
    We rely on Python Context Manager or ``With`` class(`include/tvm/support/with.h`_) to exit ``PassContext`` safely

内建 Instrument
^^^^^^^^^^^^^^^^^^^

有几个内置的 Instrument。那些标记了 *TODO* 的还没有实现。

- PassTimingInstrument (see `src/ir/instrument.cc`_)

  * Profile the execution time of passes.

- PrintIRBefore(TODO)

  * Print the IR module before the pass transforms it. :py:func:`tvm.transform.PrintIR`
    can also serve this purpose if we insert it around passes. However,
    with the ``PassInstrument``, we don't need to modify the sequence of passes.

- PrintAfter(TODO)

  * Print the IR module after the pass transforms it.

Python 前端
~~~~~~~~~~~~~~~

前端只需要一些简单的 API。
例如，可以为用户提供以下 API 来创建和执行 pass（完整实现见 `python/tvm/relay/transform/transform.py`_ 和 `python/tvm/ir/transform.py`_）。
后端接收信息并决定使用哪个函数来创建 Pass 对象。

PassContext
^^^^^^^^^^^

Python 前端通过覆盖 ``__enter__`` 和 ``__exit__`` 为 ``PassContext`` 提供了包装器来启用 ``with`` 语法。
为用户提供了 ``current`` 静态方法来获取在一定范围内正在使用的上下文。

.. code:: python

    @tvm._ffi.register_object("transform.PassContext")
    class PassContext(tvm.runtime.Object):
        def __enter__(self):
            _transform.EnterPassContext(self)
            return self

        def __exit__(self, ptype, value, trace, config):
            _transform.ExitPassContext(self)

        @staticmethod
        def current():
            """Return the current pass context."""
            return _transform.GetCurrentPassContext()

``PassContext`` 用于配置编译选项，包括优化级别和所需/禁用的 pass。
它还可以使用配置字典，以便不同的 pass 可以方便地获取 pass 的数据，例如回退设备信息和循环展开的 step/depth 等。
为了能够获取所需的配置，key 必须通过 ``TVM_REGISTER_PASS_CONFIG_OPTION`` 进行注册。例如，循环展开 pass 使用以下代码

.. code:: c++

    TVM_REGISTER_PASS_CONFIG_OPTION("tir.UnrollLoop", UnrollLoopConfig);

请参阅 `src/tir/transforms/unroll_loop.cc`_ 了解更多细节。

Pass 对象
^^^^^^^^^^^^

``Pass`` 是所有 pass 对象的基类。这里的所有方法都只是在后端实现的简单包装器。
它们是为用户定义的，以便方便地与 Python 中的基类交互。
在 pass 基类中只定义了 ``__call__``，以使子类成为可调用对象，以便它们可以轻松调用（例如 ``pass_xx(arg)``）执行。

.. code:: python

    @register_relay_node
    class Pass(RelayNode):
       def __call__(self, mod):
           return _transform.RunPass(self, mod)

提供了一些辅助 API，以支持从 Python 前端轻松创建 pass，并让 pass infra 控制执行。
例如， ``module_pass`` 、 ``function_pass`` 和 ``sequential`` 被提供给用户，以便他们可以定制自己的 pass 或 pass 管道。

对于所有在 C++ 后端实现的 pass，在 `python/tvm/ir/transform.py`_ 和 `python/tvm/relay/transform/transform.py`_ 中提供了相应的 Python API。
例如，const 折叠有如下的 Python API：

.. code:: python

    def FoldConstant():
        return _transform.FoldConstant()

用户可以借助装饰器创建 pass，如下所示：

.. code:: python

    @relay.transform.module_pass(opt_level=2)
    def transform(mod, ctx):
       tp = relay.TensorType((10,), "float32")
       x = relay.var("x", tp)
       gv = relay.GlobalVar("abs")
       func = relay.Function([x], relay.abs(x))
       new_mod = tvm.IRModule({gv: func})
       new_mod.update(mod)
       return new_mod

   module_pass = transform
   assert isinstance(module_pass, transform.ModulePass)
   assert module_pass.info.opt_level == 2

这里的 ``transform`` 函数向输入模块添加了 ``abs`` 函数，但它也可以是模块级的任何定制优化。
创建这个 ``module_pass`` 之后，用户可以将它应用到任何 Relay 模块上。
例如，可以构建空模块，并应用此传递来添加 ``abs`` 函数。

.. code:: python

    mod = tvm.IRModule()
    mod = module_pass(mod)

相应地，也为 ``function_pass`` 提供了这样的功能。例如，函数级 pass 的例子可以这样写：

.. code:: python

    @relay.transform.function_pass(opt_level=1)
    class TestReplaceFunc:
       def __init__(self, new_func):
          self.new_func = new_func
          def transform_function(self, func, mod, ctx):
             # Just for demo purposes
             # Transform func to new_func
             return self.new_func

    x = relay.var("x", shape=(10, 20))
    f1 = relay.Function([x], x)
    f2 = relay.Function([x], relay.log(x))
    # fpass is now a special pass that replaces every
    # function to f1
    fpass = TestReplaceFunc(f1)
    # Now every function in input_mod is replaced by f1
    res_mod = fpass(input_mod)

或者，用户也可以直接注册 pass，而不使用装饰器，然后调用它。
关于如何定制自己的优化管道和调试 Relay 和 tir pass 的更多示例，请参阅 `use pass infra`_ 教程。


.. _pass_instrument_py_frontend:

Pass Instrument
^^^^^^^^^^^^^^^

你可以通过在实现以下方法的类上使用 ``pass_instrument`` 装饰器(`python/tvm/ir/instrument.py`_)来实现 ``PassInstrument`` 。
注意，建议使用 ``pass_instrument`` 装饰器来实现 ``PassInstrument``，而不是重写或子类化。

- ``enter_pass_ctx``

  * 该方法在进入 ``PassContext`` 时运行。

- ``exit_pass_ctx``

  * 该方法在退出 ``PassContext`` 时运行。

- ``should_run``

  * 此方法在执行 pass 之前运行，返回布尔值，指示是否应该运行 pass。

- ``run_before_pass``

  * 如果要运行 pass，这个方法会在 pass 执行之前运行。

- ``run_after_pass``

  * 此方法在执行 pass 之后立即运行。

``PassInstrument`` 实例可以通过 :py:class:`tvm.transform.PassContext` 中的 ``instruments`` 参数进行注册。

`use pass instrument`_ 教程提供了如何用 Python API 实现 ``PassInstrument`` 的例子。

.. _pass_instrument_overriden:

在 Current PassContext 中覆写 Instruments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

提供 ``override_instruments`` 方法来覆盖当前 ``PassContext`` 的 ``instruments``。
例如，如果 pass 在运行时没有显式地创建新的 ``PassContext``，仍然可以通过以下方式将 ``PassInstrument`` 注册到全局 ``PassContext``：

.. code:: python

    cur_pass_ctx = tvm.transform.PassContext.current()
    # override PassInstrument instances
    cur_pass_ctx.override_instruments([pass_inst])
    mod = pass_seq(mod)
    result = pass_inst.get_result()

注意，当调用 ``override_instruments`` 时，会调用旧 ``PassInstrument`` 实例的 ``exit_pass_ctx`` 方法。
然后调用新的 ``PassInstrument`` 的 ``enter_pass_ctx`` 方法。

.. _Sequential: https://pytorch.org/docs/stable/nn.html?highlight=sequential#torch.nn.Sequential

.. _Block: https://mxnet.apache.org/api/python/docs/api/gluon/block.html#gluon-block

.. _include/tvm/ir/transform.h: https://github.com/apache/tvm/blob/main/include/tvm/ir/transform.h

.. _include/tvm/support/with.h: https://github.com/apache/tvm/blob/main/include/tvm/support/with.h

.. _src/relay/ir/transform.cc: https://github.com/apache/tvm/blob/main/src/relay/ir/transform.cc

.. _src/ir/transform.cc: https://github.com/apache/tvm/blob/main/src/ir/transform.cc

.. _src/ir/instrument.cc: https://github.com/apache/tvm/blob/main/src/ir/instrument.cc

.. _src/relay/transforms/fold_constant.cc: https://github.com/apache/tvm/blob/main/src/relay/transforms/fold_constant.cc

.. _python/tvm/relay/transform/transform.py: https://github.com/apache/tvm/blob/main/python/tvm/relay/transform/transform.py

.. _include/tvm/relay/transform.h: https://github.com/apache/tvm/blob/main/include/tvm/relay/transform.h

.. _python/tvm/ir/transform.py: https://github.com/apache/tvm/blob/main/python/tvm/ir/transform.py

.. _python/tvm/ir/instrument.py: https://github.com/apache/tvm/blob/main/python/tvm/ir/instrument.py

.. _src/tir/transforms/unroll_loop.cc: https://github.com/apache/tvm/blob/main/src/tir/transforms/unroll_loop.cc

.. _use pass infra: https://github.com/apache/tvm/blob/main/tutorials/dev/use_pass_infra.py

.. _use pass instrument: https://github.com/apache/tvm/blob/main/tutorials/dev/use_pass_instrument.py
