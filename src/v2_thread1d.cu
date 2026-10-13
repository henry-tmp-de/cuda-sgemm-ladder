// v2_thread1d.cu —— 1D thread tiling
//
// v1 的问题：一个线程只算 1 个输出元素，每轮 k 循环里
//   2 次全局读 + 32 次 shared 读 + 1 次 FMA，计算密度极低，
//   大部分时间在等 shared memory 和 __syncthreads。
//
// 这一步的改动：让**一个线程负责同一列的 TM 个元素**。
//   对每个 k，B 的一个元素被寄存器接住，然后和 A 的 TM 个元素做乘加。
//   B 的 shared 读次数从 TM 次降到 1 次 —— 这就是“复用”。
//
// 为什么叫 1D：线程只在行方向展开（一列里的 TM 个元素），列方向不动。
#include "kernels.h"

namespace gemm {

template <int BM, int BN, int BK, int TM>
__global__ void sgemm_v2_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M,
                                int N,
                                int K) {
  static_assert(BM % TM == 0, "BM must be divisible by TM");
  constexpr int THREADS = (BM / TM) * BN;

  __shared__ float SA[BM][BK];
  __shared__ float SB[BK][BN];

  const int tx = threadIdx.x;               // 列方向（BN 方向）
  const int ty = threadIdx.y;               // 行方向的一"段"，每段 TM 行
  const int tid = ty * blockDim.x + tx;
  const int col = blockIdx.x * BN + tx;
  const int row0 = blockIdx.y * BM + ty * TM;

  float sum[TM];
#pragma unroll
  for (int m = 0; m < TM; ++m) sum[m] = 0.0f;

  for (int t = 0; t < K; t += BK) {
    // 加载 A tile (BM x BK)：用 tid 做网格跨步，保证合并访存
#pragma unroll
    for (int i = tid; i < BM * BK; i += THREADS) {
      const int r = i / BK;
      const int c = i % BK;
      const int gr = blockIdx.y * BM + r;
      const int gc = t + c;
      SA[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }
    // 加载 B tile (BK x BN)
#pragma unroll
    for (int i = tid; i < BK * BN; i += THREADS) {
      const int r = i / BN;
      const int c = i % BN;
      const int gr = t + r;
      const int gc = blockIdx.x * BN + c;
      SB[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      // B 的这一个元素读一次进寄存器，被 TM 次 FMA 复用
      const float b_val = SB[k][tx];
#pragma unroll
      for (int m = 0; m < TM; ++m) {
        sum[m] += SA[ty * TM + m][k] * b_val;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < TM; ++m) {
    const int gr = row0 + m;
    if (gr < M && col < N) C[gr * N + col] = sum[m];
  }
}

void launch_v2_thread1d(const float* A, const float* B, float* C, int M, int N, int K) {
  // 走过的弯路（见 docs/02-优化过程复盘.md）：
  //   一开始配 BM=128,BN=64,TM=8 -> block(64,16)=1024 线程，结果 launch 直接失败：
  //   "too many resources requested for launch"。
  //   原因：1024 线程/块时每线程最多只能有 65536/1024 = 64 个寄存器，
  //   而这个 kernel 的地址计算 + sum[8] 累加器超过了 64 个，于是根本起不来。
  //   （现象很隐蔽：耗时约等于 0、结果全是 0，却不报错。）
  //   改成 512 线程/块后，每线程可用寄存器翻倍到 128 个，问题消失。
  constexpr int BM = 128, BN = 64, BK = 16, TM = 16;
  const dim3 block(BN, BM / TM);  // (64, 8) = 512 线程
  const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_v2_kernel<BM, BN, BK, TM><<<grid, block>>>(A, B, C, M, N, K);
}

}  // namespace gemm
