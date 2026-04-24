# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

cmake_minimum_required(VERSION 3.18)
project(flexloopy)

option(TVM_FFI_EXT_FROM_SOURCE "Build tvm_ffi from source, useful for cross compilation." OFF)
option(TVM_FFI_EXT_SHIP_DEBUG_SYMBOLS "Ship debug symbols" ON)

find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED
)
if (TVM_FFI_EXT_FROM_SOURCE)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --sourcedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE tvm_ffi_ROOT
  )
  message(STATUS "Building tvm_ffi from source: ${tvm_ffi_ROOT}")
  add_subdirectory(${tvm_ffi_ROOT} tvm_ffi)
else ()
  # tvm_ffi is installed via pip. Try to find tvm_ffi automatically
  find_package(tvm_ffi CONFIG REQUIRED)
endif ()

# use the projects as usual
add_library(flexloopy SHARED src/extension.cc)
# Disable stub generation during build to avoid circular dependency
# Stub generation requires importing tvm_ffi which triggers ffi.Enum registration
# which fails because the C++ library isn't fully loaded yet during build
tvm_ffi_configure_target(flexloopy STUB_DIR "./python" STUB_INIT OFF)

if (TVM_FFI_EXT_SHIP_DEBUG_SYMBOLS)
  # ship debugging symbols for backtrace on macos
  tvm_ffi_add_prefix_map(flexloopy ${CMAKE_CURRENT_SOURCE_DIR})
  tvm_ffi_add_apple_dsymutil(flexloopy)
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
    DESTINATION .
    FILES_MATCHING
    PATTERN "*.dSYM"
  )
endif ()

install(TARGETS flexloopy DESTINATION .)
tvm_ffi_install(flexloopy DESTINATION .)
