// v1_smem.cu —— shared memory block tiling
//
// 相比 v0 唯一的改动：把 A、B 的一小块先搬进 shared memory，块内所有线程
// 从 shared memory 反复读，而不是各自去全局内存捞。
//
// 为什么这一步是必须的：
//   v0 里算 C 的每个元素就要读 A 的一整行 + B 的一整列，算术强度只有 0.25。
//   搬进 shared memory 后，一个 BM x BK 的 tile 被 BN 个线程共用，
//   一个 BK x BN 的 tile 被 BM 个线程共用，全局访存量降到 1/BM + 1/BN 量级。
//
// 但注意：这一步的收益在**大矩阵**上才明显。dlog 那篇里 v1-smem 只比 v0 快一点
// （5.65 vs 4.46 TFLOPS），原因见 docs/02-优化过程复盘.md：
// 32x32 的 tile 太小，每个 block 每轮只搬 2KB，同步开销（两次 __syncthreads）
// 几乎把复用的收益吃掉了。
#include "kernels.h"

namespace gemm {

__global__ void sgemm_v1_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M,
                                int N,
                                int K) {
  constexpr int BM = 32, BN = 32, BK = 32;
  __shared__ float SA[BM][BK];
  __shared__ float SB[BK][BN];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * BM + ty;
  const int col = blockIdx.x * BN + tx;

  float sum = 0.0f;

  for (int t = 0; t < K; t += BK) {
    // A 的加载：同一 warp 内 tx 连续 -> 读 A 的连续地址 -> 合并访存。
    // 越界补 0，这样后面不用额外判断。
    SA[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.0f;
    // B 的加载：同一 warp 内 tx 连续 -> 读 B 的连续地址 -> 合并访存。
    // 注意这里 SB 的第 2 维是 N 方向的列，tx 直接对应，所以天然连续。
    SB[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      sum += SA[ty][k] * SB[k][tx];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

void launch_v1_smem(const float* A, const float* B, float* C, int M, int N, int K) {
  const dim3 block(32, 32);
  const dim3 grid((N + 31) / 32, (M + 31) / 32);
  sgemm_v1_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

}  // namespace gemm
