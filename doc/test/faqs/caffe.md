# `caffe` 常见错误

## ImportError: No module named "google"

解决方法：

```bash
pip install -i https://mirrors.aliyun.com/pypi/simple/ google-cloud
pip install -i https://mirrors.aliyun.com/pypi/simple/ google-cloud-vision
```

如果还是有 bug，则：

```bash
pip install --upgrade google-api-python-client
pip install google.cloud.bigquery
pip install google.cloud.storage
```

## TypeError: Couldn't build proto file into descriptor pool: duplicate file name caffe/proto/caffe.proto

```bash
pip install protobuf==3.20.3
```
