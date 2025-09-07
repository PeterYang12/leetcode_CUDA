#include <cuda_runtime.h>
#include <stdio.h>

template <typename T>
__global__ void naive_reduce(T *data, int N)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N && tid != 0)
    {
        atomicAdd(&data[0], data[tid]);
    }
}

template <typename T>
void init_vector(T *vec, int N, T val)
{
    for (int i = 0; i < N; i++)
    {
        vec[i] = val;
    }
}

int main()
{
    int N = 1024;
    size_t size = sizeof(float) * N;
    float *host_in = (float *)malloc(size);
    init_vector<float>(host_in, N, 1.0f);
    float *d_in;
    cudaMalloc((void **)&d_in, size);
    cudaError_t err = cudaMemcpy(d_in, host_in, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(host_in);
        return -1;
    }
    // launch kernel
    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
    naive_reduce<<<gridDim, blockDim>>>(d_in, N);

    cudaDeviceSynchronize();

    float *host_out = (float *)malloc(sizeof(float));
    cudaMemcpy(host_out, d_in, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Result: %f\n", *host_out);
    cudaFree(d_in);
    free(host_out);
    free(host_in);

    return 0;
}