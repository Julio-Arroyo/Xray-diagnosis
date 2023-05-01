cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_C_COMPILER "/home/jarroyoi/GCC-12.2.0/bin/gcc")
set(CMAKE_CXX_COMPILER "/home/jarroyoi/GCC-12.2.0/bin/g++")

project(train)

set(CMAKE_PREFIX_PATH /groups/CS156b/libtorch_preABI/libtorch)
set(CUDA_INCLUDE_DIRS /software/Modules/modulefiles/libraries/cuda/12.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(Torch REQUIRED)

include_directories(include)

add_executable(${PROJECT_NAME} "src/scalarCNN.cpp" "src/utils.cpp" "src/train.cpp")

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
