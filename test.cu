#include <iostream>
#include <cstdio>

__global__ void hi_gpu() {
    printf("Hello World from GPU!\n");
}

int main(int argc, char **argv) {
    printf("Hello World from CPU!\n");

    hi_gpu<<<1, 10>>>();

    const cudaError_t err_t = cudaDeviceReset();
    const std::string err_s   = cudaGetErrorString(err_t);
    if (err_t != cudaSuccess) {
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);
        fprintf(stderr, "code: %d, reason: %s\n", err_t, err_s.c_str());
        exit(1);
    }
    return 0;
}