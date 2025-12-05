import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from google.protobuf import text_format
import caffe_pb2 as pb2

@dataclass
class BatchNormParams:
    mean: np.ndarray          # 批处理均值（已缩放）
    var: np.ndarray           # 批处理方差（已缩放）
    eps: float                # 防止除零的小常数
    inv_std: np.ndarray       # 标准差的倒数 (1 / sqrt(var + eps))


@dataclass
class ScaleParams:
    gamma: np.ndarray         # 缩放因子
    beta: np.ndarray          # 偏移量
    has_bias: bool            # 是否包含偏移量参数


def get_bn_params(init_dict: Dict[str, Any], bn_layer: Any) -> BatchNormParams:
    blobs = init_dict[bn_layer.name].blobs
    inv_std = np.asarray(blobs[2].data, dtype=np.float32)
    inv_std = 1.0 / inv_std if inv_std.size else 1.0
    
    return BatchNormParams(
        mean=np.asarray(blobs[0].data, dtype=np.float32) * inv_std,
        var=np.asarray(blobs[1].data, dtype=np.float32) * inv_std,
        eps=bn_layer.batch_norm_param.eps,
        inv_std=inv_std
    )


def get_scale_params(init_dict: Dict[str, Any], scale_layer: Any) -> ScaleParams:
    blobs = init_dict[scale_layer.name].blobs
    gamma = np.asarray(blobs[0].data, dtype=np.float32)
    has_bias = scale_layer.scale_param.bias_term
    beta = np.asarray(blobs[1].data, dtype=np.float32) if has_bias else np.zeros_like(gamma)
    
    return ScaleParams(gamma, beta, has_bias)


def fuse_layers(init_dict: Dict[str, Any], bn_layer: Any, scale_layer: Any) -> List[np.ndarray]:
    bn_params = get_bn_params(init_dict, bn_layer)
    scale_params = get_scale_params(init_dict, scale_layer)
    std_inv = 1.0 / np.sqrt(bn_params.var + bn_params.eps)
    
    return [bn_params.mean, bn_params.var, bn_params.eps,
            scale_params.gamma, scale_params.beta, std_inv]


def _fuse_network(layers: List[Any], init_dict: Dict[str, Any], new_bn: Dict[str, Any]) -> Tuple[List[Any], Dict[str, str]]:
    new_layers = []
    pending_bn = None  # 等待匹配Scale层的BatchNorm层
    changed = {}       # 记录被融合的Scale层 -> 对应BatchNorm层的映射

    for i, layer in enumerate(layers):
        # 直接处理输入层
        if layer.type == "Input":
            new_layers.append(layer)
            continue

        # 情况1：当前是BatchNorm且下一层是Scale -> 暂存BN层等待处理
        if (layer.type == "BatchNorm" 
            and i + 1 < len(layers) 
            and layers[i + 1].type == "Scale"):
            pending_bn = layer
            continue

        # 情况2：当前是Scale且有暂存的BN层 -> 执行融合
        if layer.type == "Scale" and pending_bn is not None:
            new_bn[pending_bn.name] = fuse_layers(init_dict, pending_bn, layer)
            new_layers.append(pending_bn)  # 保留融合后的BN层
            changed[layer.name] = pending_bn.name  # 记录名称映射
            pending_bn = None  # 重置暂存
            continue

        # 情况3：独立的BN/Scale层（无法融合）
        if layer.type in ("BatchNorm", "Scale"):
            new_layers.append(layer)
            pending_bn = None  # 避免错误匹配
            continue

        # 情况4：其他类型的层 -> 更新输入连接
        layer.bottom[:] = [changed.get(bottom, bottom) for bottom in layer.bottom]
        new_layers.append(layer)

    return new_layers, changed


def fuse_network(init_net: pb2.NetParameter, predict_net: pb2.NetParameter) -> Tuple[pb2.NetParameter, pb2.NetParameter]:
    # 1. 准备初始化参数字典（兼容不同protobuf格式）
    use_layer_field = bool(init_net.layer)  # 判断用layer还是layers字段
    init_layers = init_net.layer if use_layer_field else init_net.layers
    init_layer_dict = {il.name: il for il in init_layers}

    # 2. 执行层融合
    new_bn = {}
    new_layers, changed = _fuse_network(predict_net.layer, init_layer_dict, new_bn)

    # 3. 更新预测网络结构
    predict_net.ClearField('layer')
    predict_net.layer.extend(new_layers)

    # 4. 处理初始化网络（过滤+更新参数）
    # 4.1 过滤掉已融合的Scale层
    changed_names = set(changed.keys())
    remaining_layers = [l for l in init_layers if l.name not in changed_names]
    
    # 4.2 更新保留的BN层参数
    for layer in remaining_layers:
        if layer.name in new_bn:
            mean, var, eps, gamma, beta, std_inv = new_bn[layer.name]
            layer.blobs[0].data[:] = mean.ravel().tolist()
            layer.blobs[1].data[:] = var.ravel().tolist()
            layer.blobs[2].data[:] = (gamma * std_inv).ravel().tolist()

    # 5. 重建初始化网络
    init_net.Clear()
    init_net.name = predict_net.name
    (init_net.layer.extend(remaining_layers) if use_layer_field 
     else init_net.layers.extend(remaining_layers))

    return init_net, predict_net
    

if __name__ == "__main__":
    from caffe_utils import unity_struct
    proto_file = "ResNet-50-deploy.prototxt"
    blob_file = "ResNet-50-model.caffemodel"
    # 加载网络定义和参数
    init_net = pb2.NetParameter()
    predict_net = pb2.NetParameter()
    with open(proto_file, 'r') as f:
        text_format.Merge(f.read(), predict_net)
    with open(blob_file, 'rb') as fp:
        init_net.ParseFromString(fp.read())
    predict_net = unity_struct(predict_net)
    init_net, predict_net = fuse_network(init_net, predict_net)
    with open("test.prototxt", "w") as fp: # 保存网络结构
        fp.write(text_format.MessageToString(predict_net))
    with open("test.caffemodel", "wb") as fp: # 保存网络权重
        fp.write(init_net.SerializeToString())