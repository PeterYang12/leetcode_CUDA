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

__global__ void reduce_kernel_bank_conflict(const float *data, float *result) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = global_id >= N ? 0 : data[global_id];
    __syncthreads();

    // 1.块内归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 2.块间归约
    if (tid == 0) {
        atomicAdd(&result[0], sdata[0]);
    }
}


__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        // 1.1 warp内归约，无需同步
        // 1.2 每个线程和它的offset线程进行归约
        // 1.3 warpSize=32
        // 1.4 offset=16,8,4,2,1
        // 1.5 0-15和16-31归约，0-7和8-15归约，0-3和4-7归约，0-1和2-3归约，0和1归约
        // val += __shfl_down_sync(0xffffffff, val, offset);
        unsigned mask = __activemask();
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__global__ void reduce_kernel_warp(const float *data, float *result) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = global_id < N ? data[global_id] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // 1.1 s<32,1g个warp活跃，无需同步
    if (tid < 32) {
        float val = sdata[tid]; // load 到寄存器
        val = warpReduceSum(val);    // 寄存器内 warp 归约
        if (tid == 0)
            atomicAdd(&result[0], val);
    }
}

__global__ void reduce_kernel_warp_shuffle(const float *data, float *result) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = global_id >= N ? 0 : data[global_id];
    __syncthreads();

    if (BLOCK_SIZE >= 1024) {
        if (tid < 512) {
            sdata[tid] += sdata[tid + 512];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 64) {
        if (tid < 32) {
            sdata[tid] += sdata[tid + 32];
        }
        __syncthreads();
    }

    if (tid < 32) {
        float val = sdata[tid];
        val = warpReduceSum(val);
        if (tid == 0) {
            atomicAdd(&result[0], val);
        }
    }
}

__global__ void reduce_kernel_thread_coarsening(const float *data, float *result) {

    __shared__ float sdata[BLOCK_SIZE];

    int grid_size = BLOCK_SIZE * gridDim.x;
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = 0;
    float sum = sdata[tid];

    while (global_id < N) {
        sum += data[global_id];
        global_id += grid_size;
    }
    sdata[tid] = sum;

    __syncthreads();

    if (BLOCK_SIZE >= 1024) {
        if (tid < 512) {
            sdata[tid] += sdata[tid + 512];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        float val = sdata[tid] + sdata[tid + 32];
        val = warpReduceSum(val);
        if (tid == 0) {
            atomicAdd(&result[0], val);
        }
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
    host_out[0] = 0;
    cudaMemcpy(host_out, d_out, sizeof(T), cudaMemcpyDeviceToHost);
    printf("Result: %f\n", *host_out);
    cudaFree(d_in);
    cudaFree(d_out);
    free(host_out);
    free(host_in);
}

int main() {
    cout << "launch kernel: reduce2----" << endl;
    test_reduce<float, reduce2<float>>();
    cout << "launch kernel: reduce3----" << endl;
    test_reduce<float, reduce3<float>>();
    cout << "launch kernel: reduce_kernel_bank_conflict-----" << endl;
    test_reduce<float, reduce_kernel_bank_conflict>();
    cout << "launch kernel: reduce_kernel_warp----" << endl;
    test_reduce<float, reduce_kernel_warp>();
    cout << "launch kernel: reduce_kernel_warp_shuffle----" << endl;
    test_reduce<float, reduce_kernel_warp_shuffle>();
    cout << "launch kernel: reduce_kernel_thread_coarsening----" << endl;
    test_reduce<float, reduce_kernel_thread_coarsening>();
    return 0;
}