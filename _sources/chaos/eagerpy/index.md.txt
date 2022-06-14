# EagerPy

[EagerPy](https://eagerpy.jonasrauber.de/) 是编写使用 [PyTorch](https://pytorch.org/)、[TensorFlow](https://www.tensorflow.org/)、[JAX](https://github.com/google/jax) 和 [NumPy](https://numpy.org/) 本机工作代码的 Python 框架。

::::{tab-set}

:::{tab-item} 本机性能
EagerPy 运算直接转换为相应的本机运算。
:::

:::{tab-item} 完全可链接的 API
所有功函数都可以作为张量对象的方法和 EagerPy 函数使用。
:::

:::{tab-item} 类型检查
借助于 EagerPy 的扩展类型注解，在运行代码之前捕获 Bug。
:::
::::


```{toctree}
:maxdepth: 3

start
convert
generic-functions
autodiff
```