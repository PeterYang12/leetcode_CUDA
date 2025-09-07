project(reduce LANGUAGES C CXX CUDA)

add_executable(naive_reduce naive_reduce.cu)
add_executable(reduce reduce.cu)