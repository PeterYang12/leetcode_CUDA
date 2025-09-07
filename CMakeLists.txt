cmake_minimum_required(VERSION 3.31)
project(leetcode_CUDA LANGUAGES C CXX CUDA)

# set global options
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set policy to suppress the warning
cmake_policy(SET CMP0146 NEW)

# Enable CUDA language support
enable_language(CUDA)
# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

add_executable(test test.cu)

set_target_properties(test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(naive)