# 部署 TVM 模块 的 Makefile 样例

IDIR = include
TVM_ROOT=$(shell cd ../..; pwd)
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core

PKG_CFLAGS = -std=c++17 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\
	-I${IDIR}\
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>

PKG_LDFLAGS = -ldl -pthread

.PHONY: clean all

all: outputs/test_alloc_array

# 定制 tvm 运行时
outputs/libtvm_runtime_pack.so: src/libtvm_runtime_pack.cc
	@mkdir -p $(@D)
	$(CXX) -c $(PKG_CFLAGS) -o $@  $^

# 使用定制的 tvm 运行时进行部署
outputs/test_alloc_array: src/test_alloc_array.cc outputs/libtvm_runtime_pack.so
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)

clean:
	rm -rf outputs
