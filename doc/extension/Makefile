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

all: outputs/libs/libtvm_ext.so outputs/libs/libtvm_plugin_module.so

# 定制 tvm 拓展运行时
# =================================================================================
outputs/libs/libtvm_ext.so: src/tvm_ext.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CXXFLAGS) -shared -o $@ $^ $(PKG_LDFLAGS) -L${TVM_ROOT}/build

# 定制 tvm 插件模块
# =================================================================================
outputs/libs/libtvm_plugin_module.so: src/plugin_module.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CXXFLAGS) -shared -o $@ $^ $(PKG_LDFLAGS) -L${TVM_ROOT}/build

outputs/libs/libtvm_NDSubClass.so: src/NDSubClass.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CXXFLAGS) -shared -o $@ $^ $(PKG_LDFLAGS) -L${TVM_ROOT}/build
clean:
	rm -rf outputs/*
