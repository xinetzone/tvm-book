cmake_minimum_required(VERSION 3.15)  # 最低 CMake 版本
project(caffe-proto)

# 查找 Protobuf 包
find_package(Protobuf REQUIRED)
if(NOT Protobuf_PROTOC_EXECUTABLE)
  message(FATAL_ERROR "未找到 protoc 编译器，请安装 Protocol Buffers")
endif()

# 定义 proto 文件和输出目录
set(PROTO_FILES ${CMAKE_CURRENT_SOURCE_DIR}/protos/caffe.proto)
set(PY_OUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python)
file(MAKE_DIRECTORY ${PY_OUT_DIR})

# 设置输出目录并生成 Python 代码
protobuf_generate_python(
  GENERATED_PY_FILES ${PROTO_FILES}
  PROTOC_OUT_DIR ${PY_OUT_DIR}
)

# 创建生成目标
add_custom_target(proto_python ALL
  DEPENDS ${GENERATED_PY_FILES}
  COMMENT "生成 Python 代码到 ${CMAKE_CURRENT_SOURCE_DIR}")

# 调试信息
message(STATUS "Protobuf 输出目录: ${PY_OUT_DIR}")
message(STATUS "protoc 路径: ${Protobuf_PROTOC_EXECUTABLE}")