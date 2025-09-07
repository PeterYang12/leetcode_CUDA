// warp_tensor_core_mapping.cu
#include <cstdio>
#include <mma.h>
using namespace nvcuda;

__global__ void inspect_wmma_mapping() {
    // 只启动一个 warp
    int lane = threadIdx.x % 32;

    // 定义一个 accumulator fragment (16x16 tile, FP32 accumulate)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag;

    // 每个线程都有一份 fragment 寄存器
    // fragment.x[] 是这个线程负责的那部分元素
    int num = frag.num_elements;

    printf("Thread %2d manages %d elements in the 16x16 tile:\n", lane, num);
    for (int i = 0; i < num; i++) {
        int row, col;
        // wmma::store_matrix_sync 可以告诉我们每个元素在 tile 中的坐标
        // 我们用一个临时数组接收
        float tmp[16*16];
        for (int j = 0; j < 16*16; j++) tmp[j] = -1.0f;
        frag.x[i] = (float)lane*100 + i; // 写入独特值方便识别
        wmma::store_matrix_sync(tmp, frag, 16, wmma::mem_row_major);

        // 找出 tmp[] 中被写入的位置
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                if (tmp[r*16+c] >= 0.0f) {
                    row = r; col = c;
                    printf("  -> (row=%2d, col=%2d)\\n", row, col);
                }
            }
        }
    }
}

int main() {
    inspect_wmma_mapping<<<1, 32>>>();  // 一个 warp
    cudaDeviceSynchronize();
    return 0;
}
