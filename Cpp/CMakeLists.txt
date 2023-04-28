cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_C_COMPILER "/home/jarroyoi/GCC-12.2.0/bin/gcc")
set(CMAKE_CXX_COMPILER "/home/jarroyoi/GCC-12.2.0/bin/g++")

project(train)
set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(train src/train.cpp)
target_link_libraries(train "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
