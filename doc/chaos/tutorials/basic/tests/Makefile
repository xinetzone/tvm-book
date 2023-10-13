# 扩展包的最小 Makefile
TVM_ROOT=$(shell cd /media/pc/data/lxw/ai/tvm; pwd)
PKG_CFLAGS = -std=c++17 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${TVM_ROOT}/3rdparty/dmlc-core/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>


PKG_LDFLAGS =-L${TVM_ROOT}/build
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
	PKG_LDFLAGS += -undefined dynamic_lookup
endif

lib/libtvm_ext.so: src/tvm_ext.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -shared -o $@ $^ $(PKG_LDFLAGS)
