// Warp Divergent

#include <cuda_runtime.h>
#include <stdio.h>
#define BLOCK_SIZE 256

template <typename T>
void init_vector(T *vec, int N, T val)
{
    for (int i = 0; i < N; i++)
    {
        vec[i] = val;
    }
}

template <typename T>
__global__ void reduce(const T *data, T *out, int N)
{
    __shared__ T sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = global_id >= N ? 0 : data[global_id];
    __syncthreads();

    // 1. 块内规约
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;
        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    // 2. 块间规约
    if (tid == 0)
    {
        atomicAdd(&out[0], sdata[0]);
    }
}

int main()
{
    int N = 1024;
    size_t size = sizeof(float) * N;
    float *host_in = (float *)malloc(size);
    init_vector<float>(host_in, N, 1.0f);
    float *d_in, *d_out;
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, sizeof(float));
    cudaError_t err = cudaMemcpy(d_in, host_in, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(host_in);
        return -1;
    }
    // launch kernel
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    reduce<<<gridDim, blockDim>>>(d_in, d_out, N);

    cudaDeviceSynchronize();

    float *host_out = (float *)malloc(sizeof(float));
    cudaMemcpy(host_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Result: %f\n", *host_out);
    cudaFree(d_in);
    cudaFree(d_out);
    free(host_out);
    free(host_in);

    return 0;
}