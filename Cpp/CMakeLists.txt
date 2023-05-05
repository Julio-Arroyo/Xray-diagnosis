cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_C_COMPILER "/home/jarroyoi/GCC-12.2.0/bin/gcc")
set(CMAKE_CXX_COMPILER "/home/jarroyoi/GCC-12.2.0/bin/g++")

# For train.cpp
project(train)
# For test.cpp
# project(test)

set(CMAKE_PREFIX_PATH /groups/CS156b/libtorch_preABI/libtorch)
set(CUDA_INCLUDE_DIRS /software/Modules/modulefiles/libraries/cuda/12.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(Torch REQUIRED)

include_directories(include)

# For train.cpp
add_executable(${PROJECT_NAME} "src/dataset.cpp" "src/utils.cpp" "src/resnet.cpp" "src/train.cpp")
# For test.cpp
# add_executable(${PROJECT_NAME} "src/resnet.cpp" "src/test.cpp")

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
