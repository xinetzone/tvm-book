# TVM 手册

[![PyPI][pypi-badge]][pypi-link]
[![GitHub issues][issue-badge]][issue-link]
[![GitHub forks][fork-badge]][fork-link]
![atom star](https://gitcode.com/flexloop/tvm-book/star/badge.svg)
[![GitHub stars][star-badge]][star-link]
[![GitHub license][license-badge]][license-link]
[![contributors][contributor-badge]][contributor-link]
[![watcher][watcher-badge]][watcher-link]
[![Binder][binder-badge]][binder-link]
[![Downloads][download-badge]][download-link]
[![PyPI - Downloads][install-badge]][install-link]
![repo size](https://img.shields.io/github/repo-size/xinetzone/tvm-book.svg)
[![Downloads Week](https://pepy.tech/badge/flexloopy/week)](https://pepy.tech/project/flexloopy)
[![Documentation Status][rtd-badge]][rtd-link]

[pypi-badge]: https://img.shields.io/pypi/v/flexloopy.svg
[pypi-link]: https://pypi.org/project/flexloopy/
[issue-badge]: https://img.shields.io/github/issues/xinetzone/tvm-book
[issue-link]: https://github.com/xinetzone/tvm-book/issues
[fork-badge]: https://img.shields.io/github/forks/xinetzone/tvm-book
[fork-link]: https://github.com/xinetzone/tvm-book/network
[star-badge]: https://img.shields.io/github/stars/xinetzone/tvm-book
[star-link]: https://github.com/xinetzone/tvm-book/stargazers
[license-badge]: https://img.shields.io/github/license/xinetzone/tvm-book
[license-link]: https://github.com/xinetzone/tvm-book/LICENSE
[contributor-badge]: https://img.shields.io/github/contributors/xinetzone/tvm-book
[contributor-link]: https://github.com/xinetzone/tvm-book/contributors
[watcher-badge]: https://img.shields.io/github/watchers/xinetzone/tvm-book
[watcher-link]: https://github.com/xinetzone/tvm-book/watchers
[binder-badge]: https://mybinder.org/badge_logo.svg
[binder-link]: https://mybinder.org/v2/gh/xinetzone/tvm-book/main
[install-badge]: https://img.shields.io/pypi/dw/flexloopy?label=pypi%20installs
[install-link]: https://pypistats.org/packages/flexloopy
[download-badge]: https://pepy.tech/badge/flexloopy
[download-link]: https://pepy.tech/project/flexloopy
[rtd-badge]: https://readthedocs.org/projects/flexloopy/badge/?version=latest
[rtd-link]: https://tvm-book.readthedocs.io/zh-cn/latest/?badge=latest

打造优质的 TVM 中文社区。

## 安装 `flexloopy`

确保系统中安装了 cmake 以及 ninja 构建系统：

```bash
# 安装 cmake
pip install cmake -i https://pypi.tuna.tsinghua.edu.cn/simple
# 安装 ninja，作为 cmake 的 generator：
conda install -c conda-forge ninja
# 安装 C/C++ 编译器
conda install -c conda-forge clang clangdev llvmdev llvm gcc gxx
```

使用 pip 安装 `flexloopy`：

```bash
pip install flexloopy
```

或者从源代码安装：

```bash
git clone https://github.com/xinetzone/tvm-book.git
cd tvm-book
pip install -ve .
```

构建和使用 whl 包：

```bash
pip wheel -v -w dist . # 构建 whl 包
pip install dist/*.whl # 安装 whl 包
# PowerShell 安装 whl 包
# Get-ChildItem -Path dist/*.whl | ForEach-Object { pip install $_ }
```
