// wmma_gemm.cu
// A minimal GEMM using NVIDIA Tensor Cores via WMMA API (FP16 inputs, FP32 accumulate)
// Requirements: CUDA 11+ and GPU with SM >= 70 (Volta/Turing/Ampere/Hopper)
// Build (e.g., for Blackwell):
//   nvcc -O3 -arch=sm_120 wmma_gemm.cu -o wmma_gemm
// Run:
//   ./wmma_gemm [M] [N] [K]
// Notes:
// - Uses A row-major (MxK), B col-major (KxN), C/D row-major (MxN)
// - M, N, K should be multiples of 16 for pure WMMA tiles. (Example pads not implemented.)
// - Computes: D = alpha * A * B + beta * C
//   (We implement beta path by a post-store add; alpha/beta are host constants.)

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
  cudaError_t _status = (call); \
  if (_status != cudaSuccess) { \
    fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_status)); \
    exit(1); \
  } \
} while(0)
#endif

// Tile sizes for WMMA (fixed 16x16x16 for FP16 MMA)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Kernel: one warp computes one 16x16 tile of output
__global__ void wmma_gemm_kernel(const half* __restrict__ A, const half* __restrict__ B,
                                 const float* __restrict__ C, float* __restrict__ D,
                                 int M, int N, int K,
                                 int lda, int ldb, int ldc, int ldd,
                                 float alpha, float beta) {
    // Each block has multiple warps; each warp handles one tile (16x16) of D
    int warpId = threadIdx.x / warpSize; // 0..warpsPerBlock-1
//   int laneId = threadIdx.x % warpSize; (void)laneId; // not used explicitly

    // Arrange warps linearly in Y: gridDim.y * warpsPerBlock covers all tile-rows
    int warpsPerBlock = blockDim.x / warpSize;

    int tileRow = blockIdx.y * warpsPerBlock + warpId; // tile index in M dimension
    int tileCol = blockIdx.x;                           // tile index in N dimension

    int row = tileRow * WMMA_M; // starting row in D
    int col = tileCol * WMMA_N; // starting col in D

    if (row >= M || col >= N) return;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> accFrag;

    wmma::fill_fragment(accFrag, 0.0f);

    // Loop over K dimension in 16-wide tiles
    for (int k = 0; k < K; k += WMMA_K) {
        // Base pointers for this tile of A and B
        const half* aTilePtr = A + row * lda + k;           // A is row-major: row stride = lda (==K)
        const half* bTilePtr = B + k + col * ldb;           // B is col-major: col stride = ldb (==K)

        // Load fragments (no bounds checks; assume multiples of 16)
        wmma::load_matrix_sync(aFrag, aTilePtr, lda);
        wmma::load_matrix_sync(bFrag, bTilePtr, ldb);

        // MMA accumulate: accFrag += aFrag * bFrag
        wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
    }

    // Scale by alpha
    for (int i = 0; i < accFrag.num_elements; ++i) {
        accFrag.x[i] *= alpha;
    }

    // Store to D tile (row-major)
    float* dTilePtr = D + row * ldd + col;
    wmma::store_matrix_sync(dTilePtr, accFrag, ldd, wmma::mem_row_major);

    // If beta != 0, add beta * C to D (elementwise) — cooperative within the warp
    if (beta != 0.0f && C != nullptr) {
        int elements = WMMA_M * WMMA_N; // 256
        for (int idx = threadIdx.x % warpSize; idx < elements; idx += warpSize) {
            int r = idx / WMMA_N; // 0..15
            int c = idx % WMMA_N; // 0..15
            int gr = row + r;
            int gc = col + c;
            if (gr < M && gc < N) {
                int outIdx = gr * ldd + gc;
                int cIdx   = gr * ldc + gc;
                D[outIdx] = D[outIdx] + beta * C[cIdx];
            }
        }
    }
}

static void cpu_gemm_ref(const std::vector<half>& A, const std::vector<half>& B,
                         const std::vector<float>& C, std::vector<float>& D,
                         int M, int N, int K, float alpha, float beta) {
    auto idxA = [K](int r, int c){ return r * K + c; }; // row-major
    auto idxB = [K,N](int r, int c){ (void)N; return r + c * K; }; // col-major
    auto idxC = [N](int r, int c){ return r * N + c; }; // row-major

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.f;
            for (int k = 0; k < K; ++k) {
                float a = __half2float(A[idxA(i,k)]);
                float b = __half2float(B[idxB(k,j)]);
                sum += a * b;
            }
            D[idxC(i,j)] = alpha * sum + beta * C[idxC(i,j)];
        }
    }
}

int main(int argc, char** argv) {
    int M = 256, N = 256, K = 256; // must be multiples of 16
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if (M % 16 || N % 16 || K % 16) {
        printf("[WARN] M, N, K should be multiples of 16. Given M=%d N=%d K=%d.\n", M, N, K);
    }

    float alpha = 1.0f, beta = 1.0f;

    printf("GEMM: D = %.1f * A(MxK) * B(KxN) + %.1f * C  with A=row-major, B=col-major, C/D=row-major\n", alpha, beta);
    printf("Sizes: M=%d N=%d K=%d\n", M, N, K);

    size_t sizeA = (size_t)M * K;
    size_t sizeB = (size_t)K * N;
    size_t sizeC = (size_t)M * N;

    std::vector<half>  hA(sizeA), hB(sizeB);
    std::vector<float> hC(sizeC), hD(sizeC), hDref(sizeC);

    // Initialize host data
    for (size_t i = 0; i < sizeA; ++i) {
        float v = static_cast<float>((i % 13) - 6) / 7.0f; // some small values
        hA[i] = __float2half(v);
    }
    for (size_t i = 0; i < sizeB; ++i) {
        float v = static_cast<float>((i % 9) - 4) / 5.0f;
        hB[i] = __float2half(v);
    }
    for (size_t i = 0; i < sizeC; ++i) {
        hC[i] = static_cast<float>((i % 7) - 3) / 11.0f;
    }

    // Device buffers
    half  *dA = nullptr, *dB = nullptr;
    float *dC = nullptr, *dD = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, sizeA * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dB, sizeB * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dC, sizeC * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dD, sizeC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeA * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeB * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hC.data(), sizeC * sizeof(float), cudaMemcpyHostToDevice));

    // Launch configuration
    int warpsPerBlock = 4;           // 4 warps = 128 threads
    dim3 block(32 * warpsPerBlock);  // 128 threads, 1D

    int tilesM = (M + WMMA_M - 1) / WMMA_M; // number of 16x16 tiles along M
    int tilesN = (N + WMMA_N - 1) / WMMA_N; // along N

    dim3 grid(tilesN, (tilesM + warpsPerBlock - 1) / warpsPerBlock);

    // Leading dimensions
    int lda = K; // row-major MxK
    int ldb = K; // col-major KxN: leading dimension is number of rows (K)
    int ldc = N; // row-major MxN
    int ldd = N; // row-major MxN

    printf("Launching grid=(%d,%d) block=(%d) warpsPerBlock=%d\n", grid.x, grid.y, block.x, warpsPerBlock);

    // Warmup
    wmma_gemm_kernel<<<grid, block>>>(dA, dB, dC, dD, M, N, K, lda, ldb, ldc, ldd, alpha, beta);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Time a few iterations
    cudaEvent_t start, stop; CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    int iters = 50;
    for (int it = 0; it < iters; ++it) {
        wmma_gemm_kernel<<<grid, block>>>(dA, dB, dC, dD, M, N, K, lda, ldb, ldc, ldd, alpha, beta);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.f; CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;

    CHECK_CUDA(cudaMemcpy(hD.data(), dD, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    // Reference on CPU
    cpu_gemm_ref(hA, hB, hC, hDref, M, N, K, alpha, beta);

    // Verify
    double max_abs_err = 0.0, mean_abs_err = 0.0;
    for (size_t i = 0; i < sizeC; ++i) {
        double err = std::abs((double)hD[i] - (double)hDref[i]);
        max_abs_err = std::max(max_abs_err, err);
        mean_abs_err += err;
    }
    mean_abs_err /= (double)sizeC;

    // Rough TFLOPs: 2*M*N*K operations
    double tflops = (2.0 * M * (double)N * K) / (avg_ms * 1e9);

    printf("Avg kernel time: %.3f ms  |  Max abs err: %.6e  Mean abs err: %.6e  |  Perf: %.2f TFLOP/s\n",
           avg_ms, max_abs_err, mean_abs_err, tflops);

    // Cleanup
    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC)); CHECK_CUDA(cudaFree(dD));
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
