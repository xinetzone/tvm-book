cmake_minimum_required(VERSION 3.15...4.0)  # 最低 CMake 版本（支持 Ninja 和 protobuf 模块）
project(caffe-proto)       # 项目名称

# 1. 查找 Protobuf 包（必须包含 protoc 编译器）
find_package(Protobuf REQUIRED)

# 检查 protoc 是否找到
if(NOT Protobuf_PROTOC_EXECUTABLE)
  message(FATAL_ERROR "未找到 protoc 编译器，请安装 Protocol Buffers")
endif()

# 2. 定义 .proto 文件路径
set(PROTO_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/protos/caffe.proto  # 待编译的 proto 文件
)

# 3. 配置生成 Python 代码的输出目录（构建目录下的 generated 文件夹）
set(PYTHON_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python)
file(MAKE_DIRECTORY ${PYTHON_OUTPUT_DIR})  # 确保目录存在

# 4. 自定义函数：调用 protoc 生成 Python 代码
function(generate_proto_python)
  foreach(proto_file ${PROTO_FILES})
    # 获取 proto 文件的文件名（不含路径）
    get_filename_component(proto_name ${proto_file} NAME_WE)
    # 生成的 Python 文件名（proto 名 + _pb2.py）
    set(py_file ${PYTHON_OUTPUT_DIR}/${proto_name}_pb2.py)
    
    # 添加自定义构建目标：通过 protoc 生成 Python 代码
    add_custom_command(
      OUTPUT ${py_file}  # 输出文件
      COMMAND ${Protobuf_PROTOC_EXECUTABLE}  # 调用 protoc
        --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/protos  # proto 文件搜索路径
        --python_out=${PYTHON_OUTPUT_DIR}  # Python 代码输出目录
        ${proto_file}  # 输入 proto 文件
      MAIN_DEPENDENCY ${proto_file}  # 依赖 proto 文件（修改后自动重新生成）
      COMMENT "生成 Python 代码: ${py_file}"  # 构建时显示的信息
    )
    
    # 将生成的 Python 文件添加到全局变量，用于后续目标依赖
    list(APPEND GENERATED_PY_FILES ${py_file})
  endforeach()
  
  # 创建一个目标，确保生成所有 Python 代码
  add_custom_target(
    proto_python ALL
    DEPENDS ${GENERATED_PY_FILES}  # 依赖所有生成的 Python 文件
  )
endfunction()

# 调用自定义函数，生成 Python 代码
generate_proto_python()

# 5. 可选：打印生成路径（方便调试）
message(STATUS "Protobuf Python 代码输出目录: ${PYTHON_OUTPUT_DIR}")
message(STATUS "protoc 路径: ${Protobuf_PROTOC_EXECUTABLE}")