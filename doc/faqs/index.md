# 常见问题

## libstdc++.so.6: version `GLIBCXX_3.4.30' not found

报错信息:
```
OSError: /media/pc/data/tmp/cache/conda/envs/py311/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /media/pc/data/lxw/ai/tvm/build/libtvm.so)
```
- 解决办法一：卸载重装 scipy 库
    ```bash
    pip uninstall scipy
    pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple 
    ```
- 解决办法二：
    ```bash
    conda install -c anaconda libstdcxx-ng
    ```
- 解决办法三：
    - 检查是否存在:
    ```bash
    strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
    ```
    结果：
    ```
    GLIBCXX_3.4
    GLIBCXX_3.4.1
    GLIBCXX_3.4.2
    GLIBCXX_3.4.3
    GLIBCXX_3.4.4
    GLIBCXX_3.4.5
    GLIBCXX_3.4.6
    GLIBCXX_3.4.7
    GLIBCXX_3.4.8
    GLIBCXX_3.4.9
    GLIBCXX_3.4.10
    GLIBCXX_3.4.11
    GLIBCXX_3.4.12
    GLIBCXX_3.4.13
    GLIBCXX_3.4.14
    GLIBCXX_3.4.15
    GLIBCXX_3.4.16
    GLIBCXX_3.4.17
    GLIBCXX_3.4.18
    GLIBCXX_3.4.19
    GLIBCXX_3.4.20
    GLIBCXX_3.4.21
    GLIBCXX_3.4.22
    GLIBCXX_3.4.23
    GLIBCXX_3.4.24
    GLIBCXX_3.4.25
    GLIBCXX_3.4.26
    GLIBCXX_3.4.27
    GLIBCXX_3.4.28
    GLIBCXX_3.4.29
    GLIBCXX_3.4.30
    GLIBCXX_3.4.31
    GLIBCXX_3.4.32
    GLIBCXX_TUNABLES
    GLIBCXX_DEBUG_MESSAGE_LENGTH
    ```
    - 建立软链接
    ```bash
    cd /media/pc/data/tmp/cache/conda/envs/py311/bin/../lib/
    mv libstdc++.so.6 libstdc++.so.6.old
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
    ```