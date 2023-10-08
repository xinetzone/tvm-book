#!/bin/bash
ODIR=outputs # 输出结果
mkdir -p ${ODIR}

# 编译
make

# 配置 C++ 运行环境
export TVM_ROOT=../..
export LD_LIBRARY_PATH=${TVM_ROOT}/build:${LD_LIBRARY_PATH}
export DYLD_LIBRARY_PATH=${TVM_ROOT}/build:${DYLD_LIBRARY_PATH}

echo "运行样例"
${ODIR}/test_alloc_array
