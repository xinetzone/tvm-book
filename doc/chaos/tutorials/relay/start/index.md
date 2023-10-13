# 快速了解 Relay

Relay IR 是纯粹的、面向表达式的语言。从计算图的角度来看，函数（{class}`~tvm.relay.function.Function` ）是计算图的子图，函数调用在子图中，将其参数替换为带有相应名称的子图中的自由变量。

```{toctree}
:maxdepth: 2

var
function
module
GraphExecutorCodegen/index
call
constant
tuple
if
let
```
