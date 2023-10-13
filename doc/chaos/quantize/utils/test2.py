import numpy as np
data_np = np.random.uniform(low=-1, high=1, size=input_shape).astype("float32")
def calibrateset():
    for _ in range(1):
        yield {"data": data_np}

print(f"当前量化配置：\n{relay.quantize.current_qconfig()}")
with tvm.transform.PassContext(opt_level=3):
    with relay.quantize.qconfig(
        calibrate_mode="kl_divergence",
        weight_scale="max",
        skip_conv_layers=[],
        skip_dense_layer=False
    ):
        print(f"当前量化配置：\n{relay.quantize.current_qconfig()}")
        qmod = relay.quantize.quantize(
            mod, origin_params, 
            calibrateset()
        )