# 部署 TVM 模块 的 Makefile 样例
# =================================================================================
IDIR = include
TVM_ROOT=$(shell cd /media/pc/data/lxw/ai/tvm; pwd)
PKG_CXXFLAGS = -std=c++17 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${TVM_ROOT}/3rdparty/dmlc-core/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\
	-I${IDIR}\
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>

PKG_LDFLAGS = -ldl -pthread
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Darwin)
	PKG_LDFLAGS += -undefined dynamic_lookup
endif

.PHONY: clean all

all: outputs/libs/libtvm_tvm_ext.so

# 定制 tvm 拓展运行时
# =================================================================================
outputs/libs/libtvm_tvm_ext.so: src/tvm_ext.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CXXFLAGS) -shared -o $@ $^ $(PKG_LDFLAGS) -L${TVM_ROOT}/build

clean:
	rm -rf outputs/*
