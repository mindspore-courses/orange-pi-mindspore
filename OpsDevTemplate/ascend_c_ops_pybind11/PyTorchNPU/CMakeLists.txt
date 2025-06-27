cmake_minimum_required(VERSION 3.16.0)
project(Ascend_C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# user-defined configuration
set(SOC_VERSION "Ascend910A" CACHE STRING "system on chip type")
# set(SOC_VERSION "Ascend310P3" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "ASCEND CANN package installation directory")
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# mkdir build && cd build && cmake .. && make && python ../add-custom_test.py 不需要下面几行
# sudo bash run.sh -v Ascend910A需要下面几行
# include_directories(/home/shared/miniconda3/envs/deepspeed_pt_py39/lib/python3.9/site-packages/torch/include/) # torch/extensions.h
# include_directories(/home/shared/miniconda3/envs/deepspeed_pt_py39/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/) # torch/all.h
# include_directories(/home/shared/miniconda3/envs/deepspeed_pt_py39/lib/python3.9/site-packages/torch_npu/include) # NPUStream.h
# link_directories(/home/shared/miniconda3/envs/deepspeed_pt_py39/lib/python3.9/site-packages/torch_npu/lib) # libtorch_npu.so
# link_directories(/home/shared/miniconda3/envs/deepspeed_pt_py39/lib/python3.9/site-packages/torch/lib) # libtorch.so
# export PYTHON_SITE_PACKAGES=$(pip show torch | grep Location | cut -f 2  -d":")
# export TORCH_INCLUDE_PATH="${PYTHON_SITE_PACKAGES}/torch/include/""

# ascendc_library use to add kernel file to generate ascendc library
ascendc_library(kernels STATIC
    add_custom.cpp
)

add_library(pybind11_lib SHARED pybind11.cpp)
target_link_libraries(pybind11_lib PRIVATE
  kernels
  torch_npu
)
execute_process(COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_PATH
)
message("TORCH_PATH is ${TORCH_PATH}")
set(ENV{ASCEND_HOME_PATH} ${ASCEND_CANN_PACKAGE_PATH})
execute_process(COMMAND python3 -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_NPU_PATH
)
message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")
target_link_directories(pybind11_lib PRIVATE
  ${TORCH_PATH}/lib
  ${TORCH_NPU_PATH}/lib
)
target_include_directories(pybind11_lib PRIVATE
  ${TORCH_NPU_PATH}/include
  ${TORCH_PATH}/include
  ${TORCH_PATH}/include/torch/csrc/api/include
)
execute_process(COMMAND python3 -m pybind11 --includes
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYBIND11_INC
)
string(REPLACE " " ";" PYBIND11_INC ${PYBIND11_INC})
target_compile_options(pybind11_lib PRIVATE
  ${PYBIND11_INC}
  -D_GLIBCXX_USE_CXX11_ABI=0
)

execute_process(COMMAND python3-config --extension-suffix
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE PYBIND11_SUFFIX
)
set_target_properties(pybind11_lib PROPERTIES
  OUTPUT_NAME add_custom${PYBIND11_SUFFIX}
  PREFIX "" SUFFIX ""
)
