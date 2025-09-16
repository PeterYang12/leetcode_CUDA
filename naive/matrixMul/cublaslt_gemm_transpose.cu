// file: cublaslt_gemm_transpose.cu
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <iostream>
#include <vector>
#include <cstring>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " (" << err << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t s = call; \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "CUBLAS error: " << s << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

int main() {
    // logical sizes: A is m x k, B is k x n, C is m x n
    const int m = 4;
    const int k = 5;
    const int n = 3;

    // prepare logical A (m x k) and B (k x n) in column-major on host
    std::vector<float> A_col_major(m * k);
    std::vector<float> B_col_major(k * n);
    std::vector<float> C_col_major(m * n, 0.0f);

    // fill A and B with simple values (column-major layout)
    for (int col = 0; col < k; ++col) {
        for (int row = 0; row < m; ++row) {
            A_col_major[row + col * m] = static_cast<float>(1 + row + col * m);
        }
    }
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < k; ++row) {
            B_col_major[row + col * k] = static_cast<float>(1 + row + col * k);
        }
    }

    // Now simulate "we pass A as transposed in memory" (i.e., memory contains A^T which is k x m)
    std::vector<float> A_storage_transposed(k * m);
    // stored(r, c) in A_storage_transposed (row r, col c, column-major) corresponds to A^T[r,c] = A[c,r]
    for (int r = 0; r < k; ++r) {
        for (int c = 0; c < m; ++c) {
            // column-major index: r + c * k for k x m matrix
            A_storage_transposed[r + c * k] = A_col_major[c + r * m];
        }
    }

    // device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(float) * k * m)); // stored as k x m
    CHECK_CUDA(cudaMalloc((void**)&d_B, sizeof(float) * k * n)); // k x n
    CHECK_CUDA(cudaMalloc((void**)&d_C, sizeof(float) * m * n)); // m x n

    CHECK_CUDA(cudaMemcpy(d_A, A_storage_transposed.data(), sizeof(float) * k * m, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B_col_major.data(), sizeof(float) * k * n, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, C_col_major.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice));

    // create cublasLt handle
    cublasLtHandle_t ltHandle;
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));

    // create operation descriptor: we want op(A) = transpose(stored A), so set TRANSA = CUBLAS_OP_T
    cublasLtMatmulDesc_t matmulDesc = nullptr;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    cudaDataType_t scaleType = CUDA_R_32F;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType));

    cublasOperation_t opA = CUBLAS_OP_T; // because memory contains A^T (k x m), we ask cublas to transpose it => logical A (m x k)
    cublasOperation_t opB = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    // create matrix layouts that describe the memory buffers (IMPORTANT: rows/cols are the shape in memory)
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    // A is stored as A^T (k x m) in memory
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, /*rows=*/k, /*cols=*/m, /*ld=*/k));
    // B stored as k x n
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, /*rows=*/k, /*cols=*/n, /*ld=*/k));
    // C stored as m x n
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, /*rows=*/m, /*cols=*/n, /*ld=*/m));

    // scalars
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // choose an algorithm via heuristic
    cublasLtMatmulPreference_t preference = nullptr;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    size_t maxWorkspaceBytes = 0;
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &maxWorkspaceBytes, sizeof(maxWorkspaceBytes)));

    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedAlgoCount = 0;
    // note: last Ddesc is the output layout (we reuse Cdesc)
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(ltHandle,
                                                matmulDesc,
                                                Adesc, Bdesc,
                                                Cdesc, Cdesc,
                                                preference,
                                                1, /* request 1 algorithm */
                                                &heuristicResult,
                                                &returnedAlgoCount));

    if (returnedAlgoCount == 0) {
        std::cerr << "No algorithm found by heuristic. Aborting." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // run matmul with chosen algo
    CHECK_CUBLAS(cublasLtMatmul(ltHandle,
                                matmulDesc,
                                &alpha,
                                d_A, Adesc,
                                d_B, Bdesc,
                                &beta,
                                d_C, Cdesc,
                                d_C, Cdesc,
                                &heuristicResult.algo,
                                nullptr, 0, /* workspace ptr and size */
                                0)); // stream 0

    // copy result back
    CHECK_CUDA(cudaMemcpy(C_col_major.data(), d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

    // print C in natural row-major view (we stored C in column-major, so element (r,c) is C_col_major[r + c*m])
    std::cout << "Result C (" << m << " x " << n << "):\n";
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            std::cout << C_col_major[row + col * m] << "\t";
        }
        std::cout << "\n";
    }

    // cleanup
    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Cdesc));
    CHECK_CUBLAS(cublasLtDestroy(ltHandle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
