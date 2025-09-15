project(tensor_core LANGUAGES C CXX CUDA)

add_executable(inspect inspect_wmma_mapping.cu)
add_executable(wmma_example wmma_gemm.cu)

