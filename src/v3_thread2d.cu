// v3_thread2d.cu —— 2D thread tiling（外积形式）
//
// v2 只让 B 被复用（1 个 B 元素 x TM 次 FMA），A 还是每算一个元素读一次。
// 这一步把 A 也接住：一个线程负责 TM x TN 的二维小方块。
//
// 关键视角的转变：从「内积」改成「外积」。
//   内积：C[i][j] = sum_k A[i][k] * B[k][j]  —— 一次算一个标量
//   外积：C[0:TM][0:TN] += A[0:TM][k] (列) * B[k][0:TN] (行)  —— 一次算一个 TMxTN 的块
//
// 于是每个 k：
//   A 读 TM 次 -> 被 TN 次 FMA 复用
//   B 读 TN 次 -> 被 TM 次 FMA 复用
//   shared memory 读次数从 TM*TN 降到 TM+TN，FMA 次数不变。
//   这是整个阶梯里对 shared memory 压力最大的一次改善。
//
// 线程布局：block 内按 (BM/TM) x (BN/TN) 排成二维网格。
#include "kernels.h"

namespace gemm {

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v3_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M,
                                int N,
                                int K) {
  static_assert(BM % TM == 0 && BN % TN == 0, "tile must divide evenly");
  constexpr int THREADS = (BM / TM) * (BN / TN);

  __shared__ float SA[BM][BK];
  __shared__ float SB[BK][BN];

  const int tx = threadIdx.x;  // 列方向第几个 TN 块
  const int ty = threadIdx.y;  // 行方向第几个 TM 块
  const int tid = ty * blockDim.x + tx;

  const int col0 = blockIdx.x * BN + tx * TN;
  const int row0 = blockIdx.y * BM + ty * TM;

  float sum[TM][TN];
#pragma unroll
  for (int m = 0; m < TM; ++m)
#pragma unroll
    for (int n = 0; n < TN; ++n) sum[m][n] = 0.0f;

  for (int t = 0; t < K; t += BK) {
#pragma unroll
    for (int i = tid; i < BM * BK; i += THREADS) {
      const int r = i / BK, c = i % BK;
      const int gr = blockIdx.y * BM + r;
      const int gc = t + c;
      SA[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }
#pragma unroll
    for (int i = tid; i < BK * BN; i += THREADS) {
      const int r = i / BN, c = i % BN;
      const int gr = t + r;
      const int gc = blockIdx.x * BN + c;
      SB[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      // 外积：把 A 的一小列和 B 的一小行取进寄存器，做一次 TMxTN 的乘加
      float a_reg[TM], b_reg[TN];
#pragma unroll
      for (int m = 0; m < TM; ++m) a_reg[m] = SA[ty * TM + m][k];
#pragma unroll
      for (int n = 0; n < TN; ++n) b_reg[n] = SB[k][tx * TN + n];
#pragma unroll
      for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int n = 0; n < TN; ++n) sum[m][n] += a_reg[m] * b_reg[n];
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < TM; ++m) {
    const int gr = row0 + m;
    if (gr >= M) continue;
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      const int gc = col0 + n;
      if (gc < N) C[gr * N + gc] = sum[m][n];
    }
  }
}

void launch_v3_thread2d(const float* A, const float* B, float* C, int M, int N, int K) {
  constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
  const dim3 block(BN / TN, BM / TM);  // (16, 16) = 256 线程
  const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_v3_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, N, K);
}

}  // namespace gemm
