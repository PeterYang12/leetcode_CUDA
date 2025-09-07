cmake_minimum_required(VERSION 3.31)
project(naive LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set policy to suppress the warning
cmake_policy(SET CMP0146 NEW)

# Enable CUDA language support
enable_language(CUDA)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)
# Link against CUDA libraries
# target_link_libraries(naive_reduce PRIVATE
#     CUDA::cudart
#     CUDA::cublas  # If using BLAS
#     CUDA::cufft   # If using FFT
# )

# Set CUDA architecture (optional)
# set_target_properties(naive_reduce PROPERTIES
#     CUDA_ARCHITECTURES "75"  # Example: Turing architecture
# )

add_executable(reduce1 reduce1.cu)
add_executable(reduce2 reduce2.cu)
add_executable(reduce3 reduce3.cu)