# CUDA 常见问题

## `subprocess` 调用 `nvcc`

运行
```bash
import subprocess
subprocess.Popen(["nvcc", "-v"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
```
报错 `[Errno 2] No such file or directory: 'nvcc'`

参考：[运行nvcc但是报错说找不到命令](https://github.com/NVIDIA/apex/issues/368)

解决方法：
```bash
export CUDA_HOME=/usr/local/cuda
```

# conda环境下使用nvcc -V报错nvcc: command not found

解决方法
```bash
conda install -c nvidia cuda-nvcc
```

## fatal error: `cuda_runtime.h`: 没有那个文件或目录

修改 `~/.bashrc`：
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cuda-12.4/include:$CPATH
export PATH=/usr/local/cuda-12.4/bin:$PATH
```
更新 `~/.bashrc`：
```bash
source ~/.bashrc
```

## conda 环境：sh: /media/pc/data/tmp/cache/conda/envs/xin/bin/../lib/libtinfo.so.6: no version information available (required by sh)

这个错误信息表明在执行某个脚本时，sh 无法找到所需的 `libtinfo.so.6` 库文件。该库文件是用于提供终端界面功能的共享库。

要解决这个问题，你可以尝试以下几个步骤：

1. 确保你的系统已经安装了 libtinfo 库。你可以使用以下命令来检查是否已安装该库：
   ```
   dpkg -l | grep libtinfo
   ```
   如果没有安装，你可以使用适合你的操作系统的包管理器进行安装。例如，在 Ubuntu 上可以使用以下命令安装：
   ```
   sudo apt-get install libtinfo6
   ```

2. 如果已经安装了 libtinfo 库，但仍然出现该错误，可能是因为库文件的路径没有正确设置。你可以尝试将包含 `libtinfo.so.6` 的目录添加到系统的库搜索路径中。可以通过编辑 `/etc/ld.so.conf` 文件或创建一个新的配置文件来实现。例如，在 Ubuntu 上可以编辑 `/etc/ld.so.conf.d/libc.conf` 文件，添加以下内容：
   ```
   /media/pc/data/tmp/cache/conda/envs/xin/lib
   ```
   然后运行以下命令更新库缓存：
   ```
   sudo ldconfig
   ```

3. 如果上述步骤都没有解决问题，可能是由于库文件的版本不兼容导致的。你可以尝试升级或降级相关的库文件版本，以使其与 sh 的要求相匹配。具体的操作方法取决于你的操作系统和使用的包管理器。

请注意，以上解决方法是基于常见的 Linux 发行版（如 Ubuntu）的情况。如果你使用的是其他操作系统，可能需要采取不同的步骤来解决该问题。

##  conda 环境：/usr/bin/ld: cannot find -lcudart_static: No such file or directory

参考：[cuda-cudart](https://anaconda.org/nvidia/cuda-cudart)
```bash
conda install nvidia/label/cuda-12.4.0::cuda-cudart
```