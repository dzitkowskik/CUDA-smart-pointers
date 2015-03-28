cmake_minimum_required(VERSION 3.1)
project(cudaSmartPointers)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
INCLUDE_DIRECTORIES("/usr/local/cuda/include/")
LINK_DIRECTORIES("/usr/local/cuda/lib/" "/usr/local/lib/")



set(SOURCE_FILES
    tests/cuda_scoped_ptr_unittest.cpp
    tests/cuda_shared_ptr_unittest.cpp
    tests/main_unittest.cpp
    tests/test_object.h
    cuda_scoped_ptr.hpp
    cuda_shared_ptr.hpp)

add_executable(cudaSmartPointers ${SOURCE_FILES})
TARGET_LINK_LIBRARIES(cudaSmartPointers gtest cudart)
