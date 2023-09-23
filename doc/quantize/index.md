# 量化

激活值量化采用逐通道方式，这意味着在每个通道上独立地对激活值进行量化。而卷积核参数共享是指在多个卷积层中使用相同的卷积核参数。如果采用逐通道方式进行激活值量化，那么每个卷积层的卷积核参数将被独立地量化，这与卷积核参数共享相悖。因此，为了契合卷积核参数共享，一般采用逐层量化的方式对激活值进行量化。

```{toctree}
auto-quantize/index
analysis
resnet18
test-auto-quantize
fake-quantization-to-integer
canonicalizations
```
