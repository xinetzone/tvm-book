# 自动量化

{func}`tvm.relay.quantize.quantize` 实现量化。

在运行量化的三个主要过程“注解”（annotate）、“校准”（calibrate）和“实现”（realize）之前，需要先进行“简化推理”、“折叠缩放轴”和“折叠常数”等准备工作。

```{toctree}
intro
prerequisite-optimize
QPartitionExpr
partition
annotate
custom
partition-conversions
```
