// cublaslt_gemm.cu
// 构建命令 (示例):
//   nvcc -O2 -std=c++17 cublaslt_gemm.cu -lcublasLt -lcublas -o cublaslt_gemm
// 可选指定架构(按你的显卡调整):
//   nvcc -O2 -std=c++17 -arch=sm_80 cublaslt_gemm.cu -lcublasLt -lcublas -o cublaslt_gemm
//
// 运行:
//   ./cublaslt_gemm [M N K]
//   例如: ./cublaslt_gemm 1024 1024 1024

#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <cassert>

#define CHECK_CUDA(expr) do {                               \
    cudaError_t _err = (expr);                               \
    if (_err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA Error %s:%d: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(_err));\
        std::exit(EXIT_FAILURE);                             \
    }                                                        \
} while (0)

#define CHECK_CUBLASLT(expr) do {                            \
    cublasStatus_t _st = (expr);                              \
    if (_st != CUBLAS_STATUS_SUCCESS) {                      \
        fprintf(stderr, "cuBLASLt Error %s:%d: %d\n",         \
                __FILE__, __LINE__, int(_st));               \
        std::exit(EXIT_FAILURE);                             \
    }                                                        \
} while (0)

static void cpu_gemm_row_major(int M, int N, int K,
                               float alpha,
                               const float* A, int lda, // row-major: lda = K
                               const float* B, int ldb, // row-major: ldb = N
                               float beta,
                               float* C, int ldc)       // row-major: ldc = N
{
    // C = alpha * A * B + beta * C
    // 全部按 row-major 解释
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.f;
            for (int k = 0; k < K; ++k) {
                // A(i,k) * B(k,j)
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

int main(int argc, char** argv)
{
    int M = 512, N = 512, K = 512;
    if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    std::cout << "GEMM: C(" << M << "x" << N << ") = A(" << M << "x" << K
              << ") * B(" << K << "x" << N << ") [row-major, float32]\n";

    // 1) 准备主机数据（row-major）
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    std::vector<float> hA(size_t(M) * K);
    std::vector<float> hB(size_t(K) * N);
    std::vector<float> hC(size_t(M) * N);
    std::vector<float> hC_ref(size_t(M) * N);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : hA) x = dist(rng);
    for (auto& x : hB) x = dist(rng);
    for (auto& x : hC) x = dist(rng);
    hC_ref = hC; // 保存一份给 CPU 校验

    float alpha = 1.0f, beta = 1.0f;

    // 2) 设备内存
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(float) * size_t(M) * K));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(float) * size_t(K) * N));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * size_t(M) * N));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(float) * size_t(M) * K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(float) * size_t(K) * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hC.data(), sizeof(float) * size_t(M) * N, cudaMemcpyHostToDevice));

    // 3) 建立 cuBLASLt 句柄
    cublasLtHandle_t ltHandle;
    CHECK_CUBLASLT(cublasLtCreate(&ltHandle));

    // 4) 创建 Matmul 描述：计算类型与缩放类型
    cublasLtMatmulDesc_t operationDesc;
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&operationDesc,
                                            CUBLAS_COMPUTE_32F,
                                            CUDA_R_32F));

    // 设定 A/B 是否转置（此例都不转置）
    cublasOperation_t opA = CUBLAS_OP_N;
    cublasOperation_t opB = CUBLAS_OP_N;
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    // 5) 创建矩阵布局（layouts），显式设置 row-major
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
            &Adesc, CUDA_R_32F, /*rows*/ M, /*cols*/ K, /*ld*/ lda));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
            &Bdesc, CUDA_R_32F, /*rows*/ K, /*cols*/ N, /*ld*/ ldb));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
            &Cdesc, CUDA_R_32F, /*rows*/ M, /*cols*/ N, /*ld*/ ldc));

    cublasLtOrder_t order = CUBLASLT_ORDER_ROW; // 行优先
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
            Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
            Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
            Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    // 6) (可选) workspace + heuristic 选算法
    size_t workspaceSize = 32ull << 20; // 32 MB
    void* dWorkspace = nullptr;
    CHECK_CUDA(cudaMalloc(&dWorkspace, workspaceSize));

    cublasLtMatmulPreference_t preference;
    CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedResults = 0;
    CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(
            ltHandle,
            operationDesc,
            Adesc,
            Bdesc,
            Cdesc,
            Cdesc,
            preference,
            1, // 请求一个可用算法
            &heuristicResult,
            &returnedResults));

    if (returnedResults == 0) {
        fprintf(stderr, "No suitable cuBLASLt Matmul algorithm found.\n");
        std::exit(EXIT_FAILURE);
    }

    // 7) 执行 Matmul
    cudaStream_t stream = 0; // 用默认流
    CHECK_CUBLASLT(cublasLtMatmul(
            ltHandle,
            operationDesc,
            &alpha,
            dA, Adesc,
            dB, Bdesc,
            &beta,
            dC, Cdesc,
            dC, Cdesc,
            &heuristicResult.algo,
            dWorkspace, workspaceSize,
            stream));

    // 等待计算完成
    CHECK_CUDA(cudaDeviceSynchronize());

    // 8) 拷回并用 CPU 校验
    CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeof(float) * size_t(M) * N, cudaMemcpyDeviceToHost));

    cpu_gemm_row_major(M, N, K, alpha, hA.data(), lda, hB.data(), ldb, beta, hC_ref.data(), ldc);

    // 计算误差
    double max_abs_err = 0.0, max_rel_err = 0.0;
    for (size_t i = 0; i < hC.size(); ++i) {
        double ref = hC_ref[i];
        double got = hC[i];
        double abs_err = std::abs(ref - got);
        double rel_err = abs_err / (std::abs(ref) + 1e-8);
        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }
    std::cout << "Max abs err: " << max_abs_err
              << " | Max rel err: " << max_rel_err << "\n";

    // 9) 资源清理
    CHECK_CUDA(cudaFree(dWorkspace));
    CHECK_CUBLASLT(cublasLtMatmulPreferenceDestroy(preference));

    CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(Adesc));
    CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(Bdesc));
    CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(Cdesc));
    CHECK_CUBLASLT(cublasLtMatmulDescDestroy(operationDesc));
    CHECK_CUBLASLT(cublasLtDestroy(ltHandle));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    std::cout << "Done.\n";
    return 0;
}
