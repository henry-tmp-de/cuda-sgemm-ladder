// v6_cpasync.cu —— cp.async 多级流水（Ampere 的硬件异步拷贝）
//
// v5 的双缓冲还有个浪费：数据必须经过寄存器中转（global -> reg -> smem），
// 每条 LDG 都要占一个寄存器，而且 reg->smem 这一步也要发指令。
//
// cp.async（sm_80 引入）让数据**从 global 直接进 shared memory，不经过寄存器**：
//     cp.async.cg.shared.global [smem], [gmem], 16;
// 一条指令搬 16 字节。这样：
//   1) 省掉寄存器中转，寄存器压力下降
//   2) 不需要先把数据取回来再写出去，指令数减半
//   3) 支持多级流水（这里用 3 级），搬运和计算重叠得更深
//
// 流水线协议由三个 PTX 原语组成：
//     cp.async.commit_group    把之前发出的所有 cp.async 打成一个「组」
//     cp.async.wait_group N    等到未完成的组数 <= N
//
// 【代价 / 取舍】cp.async 只能做「原地拷贝」，不能转置。
// 所以这一版 A 必须原样存成 [BM][BK]（v4/v5 里那个「A 转置存放以便 float4 读」
// 的技巧用不了了），计算内层读 A 只能退化成标量读。
// 这是本阶梯里一个真实的取舍，profiling 数据见 docs/02-优化过程复盘.md。
// 后续用 XOR swizzle 可以在保持 cp.async 的同时缓解 bank conflict。
#include "kernels.h"

namespace gemm {

// ---- cp.async 的 PTX 封装 ----
__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gmem_src) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// 搬运第 t 个 K-tile 到第 stage 层
#define V6_ISSUE(T, STAGE)                                                    \
  do {                                                                        \
    _Pragma("unroll") for (int i = 0; i < A_PER; ++i) {                       \
      const int idx = (tid + i * THREADS) * 4;                                \
      const int r = idx / BK, c = idx % BK;                                   \
      const int gr = a_row_base + r, gc = (T) + c;                            \
      if (gr < M && gc + 3 < K) {                                             \
        cp_async16(&SA[(STAGE)][r][c], &A[gr * K + gc]);                      \
      } else {                                                                \
        SA[(STAGE)][r][c + 0] = (gr < M && gc + 0 < K) ? A[gr * K + gc + 0] : 0.0f; \
        SA[(STAGE)][r][c + 1] = (gr < M && gc + 1 < K) ? A[gr * K + gc + 1] : 0.0f; \
        SA[(STAGE)][r][c + 2] = (gr < M && gc + 2 < K) ? A[gr * K + gc + 2] : 0.0f; \
        SA[(STAGE)][r][c + 3] = (gr < M && gc + 3 < K) ? A[gr * K + gc + 3] : 0.0f; \
      }                                                                       \
    }                                                                         \
    _Pragma("unroll") for (int i = 0; i < B_PER; ++i) {                       \
      const int idx = (tid + i * THREADS) * 4;                                \
      const int r = idx / BN, c = idx % BN;                                   \
      const int gr = (T) + r, gc = b_col_base + c;                            \
      if (gr < K && gc + 3 < N) {                                             \
        cp_async16(&SB[(STAGE)][r][c], &B[gr * N + gc]);                      \
      } else {                                                                \
        SB[(STAGE)][r][c + 0] = (gr < K && gc + 0 < N) ? B[gr * N + gc + 0] : 0.0f; \
        SB[(STAGE)][r][c + 1] = (gr < K && gc + 1 < N) ? B[gr * N + gc + 1] : 0.0f; \
        SB[(STAGE)][r][c + 2] = (gr < K && gc + 2 < N) ? B[gr * N + gc + 2] : 0.0f; \
        SB[(STAGE)][r][c + 3] = (gr < K && gc + 3 < N) ? B[gr * N + gc + 3] : 0.0f; \
      }                                                                       \
    }                                                                         \
  } while (0)

// 从第 stage 层读数据做外积。A 是 [BM][BK] 布局，只能标量读；
// B 是 [BK][BN] 布局，仍然可以 float4 读。
#define V6_COMPUTE(STAGE)                                                     \
  do {                                                                        \
    _Pragma("unroll") for (int k = 0; k < BK; ++k) {                          \
      float a_reg[TM], b_reg[TN];                                             \
      _Pragma("unroll") for (int m = 0; m < TM; ++m)                          \
          a_reg[m] = SA[(STAGE)][ty * TM + m][k];                             \
      _Pragma("unroll") for (int n = 0; n < TN; n += 4) {                     \
        const float4 v =                                                      \
            *reinterpret_cast<const float4*>(&SB[(STAGE)][k][tx * TN + n]);   \
        b_reg[n + 0] = v.x; b_reg[n + 1] = v.y;                               \
        b_reg[n + 2] = v.z; b_reg[n + 3] = v.w;                               \
      }                                                                       \
      _Pragma("unroll") for (int m = 0; m < TM; ++m)                          \
          _Pragma("unroll") for (int n = 0; n < TN; ++n)                      \
              sum[m][n] += a_reg[m] * b_reg[n];                               \
    }                                                                         \
  } while (0)

template <int BM, int BN, int BK, int TM, int TN, int STAGES>
__global__ void __launch_bounds__(((BM / TM) * (BN / TN)))
sgemm_v6_kernel(const float* __restrict__ A,
                const float* __restrict__ B,
                float* __restrict__ C,
                int M,
                int N,
                int K) {
  static_assert(BK % 4 == 0 && BN % 4 == 0 && TM % 4 == 0 && TN % 4 == 0,
                "need 16-byte alignment for cp.async");
  constexpr int THREADS = (BM / TM) * (BN / TN);
  constexpr int A_PER = BM * BK / 4 / THREADS;
  constexpr int B_PER = BK * BN / 4 / THREADS;
  static_assert((BM * BK / 4) % THREADS == 0, "A tile/threads mismatch");
  static_assert((BK * BN / 4) % THREADS == 0, "B tile/threads mismatch");

  __shared__ __align__(16) float SA[STAGES][BM][BK];  // cp.async 不能转置，只能原样存
  __shared__ __align__(16) float SB[STAGES][BK][BN];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  const int col0 = blockIdx.x * BN + tx * TN;
  const int row0 = blockIdx.y * BM + ty * TM;
  const int a_row_base = blockIdx.y * BM;
  const int b_col_base = blockIdx.x * BN;

  float sum[TM][TN];
#pragma unroll
  for (int m = 0; m < TM; ++m)
#pragma unroll
    for (int n = 0; n < TN; ++n) sum[m][n] = 0.0f;

  const int num_tiles = (K + BK - 1) / BK;

  // ---- prologue：先把前 STAGES-1 块发出去 ----
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < num_tiles) {
      V6_ISSUE(s * BK, s);
      cp_async_commit();
    }
  }

  // ---- 主循环 ----
  for (int t = 0; t < num_tiles; ++t) {
    // 等到最早发出的那一组完成，也就是 tile t 已就绪
    cp_async_wait<STAGES - 2>();
    __syncthreads();

    // 一边算 tile t，一边把 tile t+STAGES-1 发出去
    const int next = t + STAGES - 1;
    if (next < num_tiles) {
      V6_ISSUE(next * BK, next % STAGES);
      cp_async_commit();
    }

    V6_COMPUTE(t % STAGES);

    // 保证所有线程都算完这一层，下一轮才能覆盖它
    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < TM; ++m) {
    const int gr = row0 + m;
    if (gr >= M) continue;
#pragma unroll
    for (int n = 0; n < TN; n += 4) {
      const int gc = col0 + n;
      if (gc + 3 < N) {
        *reinterpret_cast<float4*>(&C[gr * N + gc]) =
            make_float4(sum[m][n], sum[m][n + 1], sum[m][n + 2], sum[m][n + 3]);
      } else {
        for (int q = 0; q < 4; ++q)
          if (gc + q < N) C[gr * N + gc + q] = sum[m][n + q];
      }
    }
  }
}

#undef V6_ISSUE
#undef V6_COMPUTE

void launch_v6_cpasync(const float* A, const float* B, float* C, int M, int N, int K) {
  constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8, STAGES = 3;
  const dim3 block(BN / TN, BM / TM);  // (16, 16) = 256 线程
  const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_v6_kernel<BM, BN, BK, TM, TN, STAGES><<<grid, block>>>(A, B, C, M, N, K);
}

}  // namespace gemm
