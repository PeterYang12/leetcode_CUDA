cmake_minimum_required(VERSION 3.31)
project(leetcode_CUDA LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(test main.cu)

set_target_properties(test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
