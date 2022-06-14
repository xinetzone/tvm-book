(ch_cpu_schedules)=
# CPU 上的算子优化

在前三章中，主要讨论了算子的功能，即如何在 TVM 中实现算子的正确函数。然而，仅仅得到正确的结果是不够的。即使 execute 正确，算子也可能表现不佳。

从本章开始，将讨论算子的性能优化。具体来说，将在本章中研究 CPU 上的算子优化，并在下一章中继续研究 GPU。

```{toctree}
:hidden:

arch
call_overhead
vector_add
broadcast_add
matmul
block_matmul
conv
packed_conv
depthwise_conv
pooling
batch_norm
```
