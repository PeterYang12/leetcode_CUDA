#include <cstdio>
#include <cuda.h>
#include <mma.h>
#include <cuda_pipeline.h>
#include <cuda/barrier>

using namespace nvcuda;

#define M 128
#define N 128
#define K 128
#define TILE 16

// Kernel: C = A * B
__global__ void matmul_tma(float *C, const float *A, const float *B) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        // ---------------- TMA copy ----------------
        cuda::barrier<cuda::thread_scope_block> bar;
        init(&bar, blockDim.x * blockDim.y);

        // A tile
        cuda::memcpy_async(&As[0][0],
                           &A[row * K + t],
                           TILE * TILE * sizeof(float),
                           bar);

        // B tile
        cuda::memcpy_async(&Bs[0][0],
                           &B[t * N + col],
                           TILE * TILE * sizeof(float),
                           bar);

        // wait for copy
        wait(bar);

        __syncthreads();
        // ------------------------------------------

        for (int k = 0; k < TILE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

int main() {
    float *A, *B, *C;
    float *dA, *dB, *dC;

    size_t size = M * K * sizeof(float);
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    // init
    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 1.0f;

    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE, TILE);
    dim3 blocks(M / TILE, N / TILE);

    matmul_tma<<<blocks, threads>>>(dC, dA, dB);

    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    printf("C[0,0] = %f\n", C[0]);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(A);
    free(B);
    free(C);

    return 0;
}
