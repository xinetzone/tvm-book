# Minimum Makefile for the extension package
# TVM_ROOT=$(shell cd /media/pc/data/4tb/lxw/workspace/tvm-dev; pwd)
TVM_ROOT=$(shell cd /media/pc/data/lxw/ai/tvm; pwd)
# TVM_ROOT=$(shell cd /media/pc/data/4tb/lxw/workspace/tvm; pwd)
PKG_CXXFLAGS = -std=c++17 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${TVM_ROOT}/3rdparty/dmlc-core/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\
	-Iinclude
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>


PKG_LDFLAGS =-L${TVM_ROOT}/build
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
	PKG_LDFLAGS += -undefined dynamic_lookup
endif

.PHONY: clean all

all: libs/libtvm_ext.so

libs/libtvm_ext.so: src/tvm_ext.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CXXFLAGS) -shared -o $@ $^ $(PKG_LDFLAGS)


clean:
	rm -rf build/* libs/* logs/*