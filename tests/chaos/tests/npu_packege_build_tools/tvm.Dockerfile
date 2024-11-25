# 镜像名称 build-tvm
FROM python:3.10
# 添加源并更新
COPY sources.list /etc/apt/sources.list
# 添加 TVM 必备包
RUN apt-get update \
    && apt-get install --assume-yes apt-utils \
    && apt-get install -y gcc g++ libtinfo-dev zlib1g-dev build-essential cmake \
    && apt install -y clang clangd llvm liblldb-dev libedit-dev libxml2-dev \
    && /usr/local/bin/python -m pip install --upgrade pip \
    && pip install decorator scipy attrs pandas toml synr \
    && apt-get install -y gcc-arm-linux-gnueabihf \
    && apt-get install -y g++-arm-linux-gnueabihf \
    && pip install nuitka invoke hatch numpy d2py

# 添加 TVM 环境
RUN echo "export TVM_HOME=/root/workspace/npu_tvm" >> /root/.bashrc \
    && echo "export PYTHONPATH=\$TVM_HOME/python:\${PYTHONPATH}" >> /root/.bashrc \
    && echo "export VTA_HW_PATH=\$TVM_HOME/3rdparty/vta-hw" >> /root/.bashrc \
    && echo "export PYTHONPATH=\$TVM_HOME/vta/python:\${PYTHONPATH}" >> /root/.bashrc

COPY tasks.py /tasks.py
