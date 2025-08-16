from tvm.relax.testing import nn
from tvm.relax import op as _op

## 保存和加载模型
with open(temp_dir/"concat.prototxt", "w") as fp: # 保存网络结构
    fp.write(text_format.MessageToString(predict_net))
with open(temp_dir/"concat.caffemodel", "wb") as fp: # 保存网络权重
    fp.write(predict_net.SerializeToString())
predict_net_load = pb2.NetParameter()
with open(temp_dir/"concat.prototxt", 'r') as fp:
    text_format.Merge(fp.read(), predict_net_load)

predict_net_load == predict_net
init_net_load = pb2.NetParameter()
with open(temp_dir/"concat.caffemodel", 'rb') as fp:
    init_net_load.ParseFromString(fp.read())

init_net_load == predict_net