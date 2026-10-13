// v0_naive.cu —— 最朴素的矩阵乘
//
// 思路：C 的每个元素之间没有数据依赖，所以把 M*N 个输出直接摊给 M*N 个线程，
//       每个线程自己跑一遍 K 长度的点积。
//
// 这个版本的唯一作用是当**基准线**。它慢在哪：
//   - 算 C 的一个元素要读 A 的一整行 (K 个数) + B 的一整列 (K 个数)
//   - 相邻线程 (col, col+1) 读 B 时，每次跨 N 个 float —— 完全没合并访存
//   - 同一份 A 的行被 N 个线程各读了一遍，零复用
// 算术强度只有 2K / (8K) = 0.25 FLOP/Byte，而 3090 需要约 38 才能喂饱。
#include "kernels.h"

namespace gemm {

__global__ void sgemm_v0_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M,
                                int N,
                                int K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = sum;
}

void launch_v0_naive(const float* A, const float* B, float* C, int M, int N, int K) {
  const dim3 block(32, 32);
  const dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  sgemm_v0_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

}  // namespace gemm
