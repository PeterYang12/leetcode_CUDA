cmake_minimum_required(VERSION 3.31)
project(tensor_core LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set policy to suppress the warning
cmake_policy(SET CMP0146 NEW)

# Enable CUDA language support
enable_language(CUDA)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)
add_executable(inspect inspect_wmma_mapping.cu)
add_executable(wmma_example wmma_gemm.cu)