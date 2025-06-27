cmake_minimum_required(VERSION 3.4...3.18)
project(my_mindspore_ops)

# ------- DEBUG添加这两行 -------
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -g")

# 1. libmy_mindspore_ops_impl.a ，SHARED动态库/STATIC静态库 源文件们
add_library(my_mindspore_ops_impl STATIC src/my_mindspore_ops_impl.cpp)
set_target_properties(my_mindspore_ops_impl PROPERTIES POSITION_INDEPENDENT_CODE ON) # -FPIC生成位置无关码

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party ${CMAKE_CURRENT_SOURCE_DIR}/../third_party) 
# 2. 生成my_mindspore_ops.cpython-312-x86_64-linux-gnu.so
# pybind11_add_module是pybind11提供的一个CMake函数，用于简化pybind11项目的生成Python拓展模块的流程。pybind11_add_module语法如下：
# pybind11_add_module(<name> [MODULE | SHARED] [EXCLUDE_FROM_ALL]
#                     [NO_EXTRAS] [THIN_LTO] [OPT_SIZE] source1 [source2 ...])
pybind11_add_module(my_mindspore_ops src/my_mindspore_ops_binder.cpp)

# 3. libmy_mindspore_ops_impl.a实现代码 链接到 动态链接库文件my_mindspore_ops.cpython-312-x86_64-linux-gnu.so  中：
target_link_libraries(my_mindspore_ops PRIVATE my_mindspore_ops_impl)
# 最后Python脚本中 import my_mindspore_ops 通过查找 my_mindspore_ops_interface.hpp 接口,接口hpp 绑定 实现my_mindspore_ops_impl.cpp
