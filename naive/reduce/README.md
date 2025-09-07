# Reduce算法的CUDA实现

基本上就是抄了一下这篇博客
https://mp.weixin.qq.com/s/92xxzkRXBjrdODjX9VjjSw

1. Naive
2. Warp Divergent
3. Bank Conflict
4. 最后一个warp优化
5. Loop unrolling
6. Warp shuffling
7. Thread coarsening