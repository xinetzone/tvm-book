import mxnet as mx

from . import set_env
import tvm
from tvm import relay


def eval_acc(model, params,
             dataset, batch_fn,
             target, device,
             logger,
             log_interval=500):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(model, target, params=params)
    # create runtime module
    m = tvm.contrib.graph_executor.GraphModule(lib["default"](device))

    # setup evaluaiton metric
    if isinstance(dataset, mx.io.io.MXDataIter):
        dataset.reset()
        batch_size = dataset.batch_size
    else:
        batch_size = 1
        
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    # Execute
    for i, batch in enumerate(dataset):
        data, label = batch_fn(batch, [mx.cpu(0)])
        m.set_input("data", tvm.nd.array(data[0].asnumpy()))
        m.run()
        out_arr = m.get_output(0)
        acc_top1.update(label, [mx.nd.array(out_arr.numpy())])
        acc_top5.update(label, [mx.nd.array(out_arr.numpy())])

        if not (i + 1) % log_interval:
            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            nsamples = (i + 1) * batch_size
            logger.info(
                "[%d samples] validation: acc-top1=%f acc-top5=%f", nsamples, top1, top5)
    logger.info("[final] validation: acc-top1=%f acc-top5=%f", top1, top5)
    return top1


