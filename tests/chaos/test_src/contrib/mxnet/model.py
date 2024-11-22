import time
import mxnet as mx
from gluoncv.model_zoo import get_model


def params_count(net, input_size):
    _params_count = 0
    net.initialize()
    _ = net(mx.nd.zeros((1, 3, input_size, input_size)))
    for _, v in net.collect_params().items():
        _params_count += v.data().size
    return _params_count


def load_deploy_model(model_prefix: str):
    '''
    Args:
        model_prefix: load static model as hybridblock.
    '''
    net = mx.gluon.SymbolBlock.imports(f'{model_prefix}-symbol.json',
                                       ['data'], f'{model_prefix}-0000.params')
    net.hybridize(static_alloc=True, static_shape=True)
    return net


def load_model(model_name, ctx, classes,
               pretrained: bool = True,
               dtype: str = 'float32',
               quantized: bool = False,
               use_se: bool = False,
               **kwargs):
    '''

    Args:
        dtype: training data type
        params_file: local parameter file to load, instead of pre-trained weight.
        quantized: use int8 pretrained model
        use_se: use SE layers or not in resnext. default is false.
    '''
    kwargs = {'pretrained': pretrained,
              'ctx': ctx,
              'classes': classes}
    if model_name.startswith('resnext'):
        kwargs['use_se'] = use_se
    net = get_model(model_name, **kwargs)
    if not pretrained:
        net.initialize()
    net.cast(dtype)
    if quantized:
        net.hybridize(static_alloc=True, static_shape=True)
    else:
        net.hybridize()
    return net


def load_model2(model_name, ctx, classes,
                dtype: str = 'float32',
                quantized: bool = False,
                params_file='',
                use_se: bool = False,
                **kwargs):
    '''

    Args:
        dtype: training data type
        params_file: local parameter file to load, instead of pre-trained weight.
        quantized: use int8 pretrained model
        use_se: use SE layers or not in resnext. default is false.
    '''
    pretrained = True if not params_file else False
    kwargs = {'pretrained': pretrained,
              'ctx': ctx,
              'classes': classes}
    if model_name.startswith('resnext'):
        kwargs['use_se'] = use_se
    net = get_model(model_name, **kwargs)
    net.cast(dtype)
    if params_file:
        net.load_parameters(params_file, ctx=ctx)
    if quantized:
        net.hybridize(static_alloc=True, static_shape=True)
    else:
        net.hybridize()
    return net


def benchmark(network, ctx,
              batch_size=64,
              image_size=224,
              num_iter=100,
              dtype='float32'):
    '''
    Examples:
        print('-----benchmark mode for model %s-----' % model_name)
        time_cost = benchmark(network=net, ctx=ctx[0], image_size=input_size, batch_size=batch_size,
                            num_iter=num_batches, datatype='float32')
        fps = (batch_size*num_batches)/time_cost
        print('With batch size %s, %s batches, inference performance is %.2f img/sec' %
            (batch_size, num_batches, fps))
    '''
    input_shape = (batch_size, 3) + (image_size, image_size)
    data = mx.random.uniform(-1.0, 1.0, shape=input_shape,
                             ctx=ctx, dtype=dtype)
    dryrun = 5
    for i in range(num_iter+dryrun):
        if i == dryrun:
            tic = time.time()
        output = network(data)
        output.asnumpy()
    toc = time.time() - tic
    return toc


def get_fps(batch_size, num_batches, time_cost):
    return (batch_size*num_batches)/time_cost
