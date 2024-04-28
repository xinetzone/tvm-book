from mxnet import gluon

from . import set_env
import tvm
from tvm.relay import quantize as qtz
import tvm.testing
from .model import get_model
from .quantize import eval_acc


def batch_fn(batch, ctx):
    data = gluon.utils.split_and_load(batch.data[0],
                                      ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch.label[0],
                                       ctx_list=ctx, batch_axis=0)
    return data, label


@tvm.testing.requires_gpu
def test_acc(config, val_data, target, device,
             logger,
             log_interval=500):
    model, params = get_model(config.model,
                              config.batch_size,
                              logger,
                              original=True)
    acc = eval_acc(model, params, val_data, batch_fn,
                   target, device, logger,
                   log_interval=log_interval)
    assert acc > config.expected_acc
    return acc


@tvm.testing.requires_gpu
def test_quantize_acc(qconfig, val_data, target, device,
                      logger,
                      skip_conv_layers=[],
                      skip_dense_layer=True,
                      log_interval=500):
    tvm_qconfig = qtz.qconfig(
        skip_conv_layers=skip_conv_layers,
        skip_dense_layer=skip_dense_layer,
        nbit_input=qconfig.nbit_input,
        nbit_weight=qconfig.nbit_input,
        global_scale=qconfig.global_scale,
        dtype_input=qconfig.dtype_input,
        dtype_weight=qconfig.dtype_input,
        dtype_activation=qconfig.dtype_output,
        debug_enabled_ops=None,
    )
    model, params = get_model(qconfig.model,
                              qconfig.batch_size,
                              logger,
                              original=False,
                              qconfig=tvm_qconfig)
    acc = eval_acc(model, params, val_data, batch_fn,
                   target, device, logger,
                   log_interval=log_interval)
    assert acc > qconfig.expected_acc
    return acc
