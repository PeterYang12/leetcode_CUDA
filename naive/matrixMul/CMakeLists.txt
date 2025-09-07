project(matrixMul LANGUAGES C CXX CUDA)

add_executable(cublaslt_gemm cublaslt_gemm.cu)

target_link_libraries(cublaslt_gemm PRIVATE CUDA::cublas)

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
