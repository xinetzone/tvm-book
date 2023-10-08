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

test: outputs/test_alloc_array outputs/main
packed: outputs/libs/libtvm_runtime_pack.so outputs/libs/libtvm_ext.so outputs/libs/libtvm_plugin.so
all: packed test

# 定制 tvm 运行时
# =================================================================================
outputs/libs/libtvm_runtime_pack.so: src/libtvm_runtime_pack.cc
	@mkdir -p $(@D)
	$(CXX) -c $(PKG_CXXFLAGS) -o $@  $^

# 使用定制的 tvm 运行时测试
# =================================================================================
outputs/test_alloc_array: src/test_alloc_array.cc outputs/libs/libtvm_runtime_pack.so
	@mkdir -p $(@D)
	$(CXX) $(PKG_CXXFLAGS) -o $@  $^ $(PKG_LDFLAGS)

# 定制 tvm 插件模块
# =================================================================================
outputs/libs/libtvm_plugin_module.so: src/plugin_module.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CXXFLAGS) -shared -o $@ $^ $(PKG_LDFLAGS) -L${TVM_ROOT}/build

# 定制 tvm 拓展运行时
# =================================================================================
outputs/libs/libtvm_ext.so: extension/src/tvm_ext.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CXXFLAGS) -shared -o $@ $^ $(PKG_LDFLAGS) -L${TVM_ROOT}/build

# 定制 tvm 拓展运行时测试
# =================================================================================
outputs/main: tests/main.cc outputs/libs/libtvm_ext.so
	@mkdir -p $(@D)
	$(CXX) $(PKG_CXXFLAGS) -o $@  $^ $(PKG_LDFLAGS)

clean:
	rm -rf outputs/*
