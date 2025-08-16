import time
import mxnet as mx
from mxnet import gluon


class Estimator:
    def __init__(self, logger, val_data, mode='image') -> None:
        self.val_data = val_data
        self.logger = logger
        self.mode = mode

    def log(self, batch_epoch, top1, top5):
        if self.mode == 'image':
            num_batch = len(self.val_data)
            self.logger.info(
                f"{batch_epoch:d} / {num_batch:d} (top1, top5) : {top1:.8f}, {top5:.8f}")
        else:
            self.logger.info(
                f"{batch_epoch:d} (top1, top5) : {top1:.8f}, {top5:.8f}")

    def test(self, network, ctx, dtype):
        acc_top1 = mx.metric.Accuracy()
        acc_top5 = mx.metric.TopKAccuracy(5)
        acc_top1.reset()
        acc_top5.reset()
        num = 0
        start = time.time()
        for i, batch in enumerate(self.val_data):
            if self.mode == 'image':
                data = gluon.utils.split_and_load(
                    batch[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(
                    batch[1], ctx_list=ctx, batch_axis=0)
            else:
                data = gluon.utils.split_and_load(
                    batch.data[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(
                    batch.label[0], ctx_list=ctx, batch_axis=0)
            batch_size = len(data)
            outputs = [network(X.astype(dtype, copy=False)) for X in data]
            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)

            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            self.log(i, top1, top5)
            num += batch_size
        end = time.time()
        speed = num / (end - start)
        self.logger.info(f'Throughput is {speed:%f} img/sec.')

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        return (1-top1, 1-top5)
