# Docker FAQs

对于中国境内 docker 无法使用的情况，可以配置代理文件 `/etc/docker/daemon.json`:
```json
{
    "data-root": "/media/pc/data/docker/lib/docker",
    "storage-driver": "overlay2",
    "registry-mirrors": [ 
    "https://docker.m.daocloud.io",
    "https://dockerproxy.com",
    "https://docker.mirrors.ustc.edu.cn",
    "http://hub-mirror.c.163.com",
    "https://docker.nju.edu.cn"
  ],
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "experimental": false,
  "features": {
    "buildkit": false
  }
}
```
重启 docker 服务：
```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```