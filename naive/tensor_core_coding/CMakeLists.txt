project(tensor_core LANGUAGES C CXX CUDA)

add_executable(inspect inspect_wmma_mapping.cu)
add_executable(wmma_example wmma_gemm.cu)
add_executable(tma_gemm tma_gemm.cu)
add_executable(wmma_tma_gemm wmma_tma_gemm.cu)

set_target_properties(wmma_tma_gemm PROPERTIES
        CUDA_ARCHITECTURES 120
)