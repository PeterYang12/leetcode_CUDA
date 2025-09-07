#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define N 1024
#define BLOCK_SIZE 256

using namespace std;

template<typename T>
__global__ void reduce2(const T *data, T *out) {
    __shared__ T sdata[256];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = global_id >= N ? 0 : data[global_id];
    __syncthreads();

    // 1. 块内规约
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // 2. 块间规约
    if (tid == 0) {
        atomicAdd(&out[0], sdata[0]);
    }
}

// 主要是将通过index的方式，将计算集中到单个warp中
template<typename T>
__global__ void reduce3(const T *data, T *out) {
    __shared__ T sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = global_id >= N ? 0 : data[global_id];
    __syncthreads();

    // 1. 块内规约
    for (int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    // 2. 块间规约
    if (tid == 0) {
        atomicAdd(&out[0], sdata[0]);
    }
}

template<typename T>
void init_vector(T *vec, T val) {
    for (int i = 0; i < N; i++) {
        vec[i] = val;
    }
}

template<typename T, void (*kernel)(const T *, T *out)>
void test_reduce() {
    size_t size = sizeof(T) * N;
    T *host_in = (T *) malloc(size);
    init_vector<T>(host_in, 1);
    T *d_in, *d_out;
    cudaMalloc((void **) &d_in, size);
    cudaMalloc((void **) &d_out, sizeof(T));
    cudaError_t err = cudaMemcpy(d_in, host_in, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(host_in);
        return;
    }
    // launch kernel
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    kernel<<<gridSize, blockSize>>>(d_in, d_out);

    cudaDeviceSynchronize();

    T *host_out = (T *) malloc(sizeof(T));
    cudaMemcpy(host_out, d_out, sizeof(T), cudaMemcpyDeviceToHost);
    printf("Result: %f\n", *host_out);
    cudaFree(d_in);
    cudaFree(d_out);
    free(host_out);
    free(host_in);
}

int main() {
    cout <<"launch kernel: reduce2" <<endl;
    test_reduce<float, reduce2<float>>();
    cout <<"launch kernel: reduce3" <<endl;
    test_reduce<float, reduce3<float>>();
    return 0;
}