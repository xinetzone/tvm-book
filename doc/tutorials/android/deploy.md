# 为 Android 目标构建模型

参考：[`how_to/deploy/android`](https://tvm.apache.org/docs/how_to/deploy/android.html)

针对 Android 目标的 Relay 编译模型可以遵循与 [android_rpc](https://github.com/apache/tvm/tree/main/apps/android_rpc) 相同的方法。下方代码将保存在 Android 目标上所需的编译输出。

```python
lib.export_library("deploy_lib.so", fcompile=ndk.create_shared)
with open("deploy_graph.json", "w") as fo:
    fo.write(graph.json())
with open("deploy_param.params", "wb") as fo:
    fo.write(runtime.save_param_dict(params))
```

`deploy_lib.so`、`deploy_graph.json`、`deploy_param.params` 将被部署到安卓目标平台。

## 安卓目标平台的 TVM 运行时

请参阅[构建适用于安卓目标的 CPU/OpenCL 版本 TVM 运行时](https://github.com/apache/tvm/blob/main/apps/android_deploy/README.md#build-and-installation)。关于从 Android Java TVM API 加载模型并执行的过程，可以参考 [Java 示例源代码](https://github.com/apache/tvm/blob/main/apps/android_deploy/app/src/main/java/org/apache/tvm/android/demo/MainActivity.java)。
