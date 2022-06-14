# d2l-tvm 教程

参考：[d2l-tvm](http://tvm.d2l.ai/index.html)

```{seealso}
访问 [roadmap](http://bit.ly/2NQ7gh3) 和 [Discussions](https://discuss.tvm.ai/t/d2l-tvm-a-tvm-introduction-book/4305) 了解更多信息。
```

```{note}
[d2l-tvm](http://tvm.d2l.ai/index.html) 项目是为那些对利用深度学习技术（尤其是模型推理）对其程序的高性能实现感兴趣，但可能还没有动手的读者准备的。假设读者以前只具有 NumPy 方面的基本知识。考虑到这一点，将从头开始解释，并在需要时介绍相关背景。然而，有经验的读者也会发现这些内容很有用。

[d2l-tvm](http://tvm.d2l.ai/index.html) 大致将内容分为三个主要部分。

- 在第一部分中，将介绍如何在各种硬件平台上实现和优化矩阵乘法、卷积等运算。这是深度学习和科学计算的基本组成部分。
- 在第二部分中，将展示如何从各种深度学习框架转换神经网络模型，并在程序级别进一步优化它们。
- 最后，将讨论如何将优化后的程序部署到各种环境中，如手机。此外，在最后，介绍一些深度学习编译器领域的最新进展。
```

```{toctree}
:maxdepth: 3

getting_started/index
expressions/index
common_operators/index
cpu_schedules/index
gpu_schedules/index
zreferences
```
