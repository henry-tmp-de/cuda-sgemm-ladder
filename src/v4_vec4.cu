// v4_vec4.cu —— 128-bit 向量化访存 + A tile 转置存储
//
// 到目前为止，所有访存都是 32-bit 的。这一步做两件事：
//
// 1) 全局内存用 float4（128-bit）加载。
//    同样搬 16 个 float，4 条 LDG.128 比 16 条 LDG.32 少发 12 条指令。
//    指令发射本身也是瓶颈，尤其在计算密度上来之后。
//
// 2) 把 A 以「转置」形式写进 shared memory，即 SA 的布局从 [BM][BK] 改成 [BK][BM]。
//    为什么？看计算内层：线程要连续读 A 的 TM 个元素 SA[ty*TM + m][k]（m 递增）。
//    在 [BM][BK] 布局下，这 TM 个元素在内存里相距 BK 个 float，**不连续**，
//    没法用 float4 读。转置成 [BK][BM] 后，SA[k][ty*TM + m] 在 m 方向上连续，
//    就能一次 float4 取 4 个。
//
//    代价：写入 SA 时变成了「按行读、按列写」的散列写入，会引入 shared memory
//    bank conflict —— 这个问题留给下一版处理（对应 dlog 那篇的后续步骤）。
#include "kernels.h"

namespace gemm {

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__(((BM / TM) * (BN / TN)))
sgemm_v4_kernel(const float* __restrict__ A,
                const float* __restrict__ B,
                float* __restrict__ C,
                int M,
                int N,
                int K) {
  static_assert(BM % TM == 0 && BN % TN == 0, "tile must divide evenly");
  static_assert(BK % 4 == 0 && TM % 4 == 0 && TN % 4 == 0, "need float4 alignment");
  constexpr int THREADS = (BM / TM) * (BN / TN);

  // A 转置存放：[k][m]，让计算内层能连续读。
  // __align__(16)：这两个数组要做 float4 读写，必须显式保证 16 字节对齐。
  __shared__ __align__(16) float SA[BK][BM];
  __shared__ __align__(16) float SB[BK][BN];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  const int col0 = blockIdx.x * BN + tx * TN;
  const int row0 = blockIdx.y * BM + ty * TM;

  float sum[TM][TN];
#pragma unroll
  for (int m = 0; m < TM; ++m)
#pragma unroll
    for (int n = 0; n < TN; ++n) sum[m][n] = 0.0f;

  for (int t = 0; t < K; t += BK) {
    // ---- 加载 A (BM x BK)，float4 读全局、转置写 smem ----
#pragma unroll
    for (int i = tid; i < BM * BK / 4; i += THREADS) {
      const int idx = i * 4;
      const int r = idx / BK;      // 行（M 方向）
      const int c = idx % BK;      // 列（K 方向）
      const int gr = blockIdx.y * BM + r;
      const int gc = t + c;
      if (gr < M && gc + 3 < K) {
        const float4 v = *reinterpret_cast<const float4*>(&A[gr * K + gc]);
        SA[c + 0][r] = v.x;
        SA[c + 1][r] = v.y;
        SA[c + 2][r] = v.z;
        SA[c + 3][r] = v.w;
      } else {
        for (int q = 0; q < 4; ++q) {
          SA[c + q][r] = (gr < M && gc + q < K) ? A[gr * K + gc + q] : 0.0f;
        }
      }
    }
    // ---- 加载 B (BK x BN)，直接 float4 读、float4 写 ----
#pragma unroll
    for (int i = tid; i < BK * BN / 4; i += THREADS) {
      const int idx = i * 4;
      const int r = idx / BN;
      const int c = idx % BN;
      const int gr = t + r;
      const int gc = blockIdx.x * BN + c;
      if (gr < K && gc + 3 < N) {
        *reinterpret_cast<float4*>(&SB[r][c]) =
            *reinterpret_cast<const float4*>(&B[gr * N + gc]);
      } else {
        for (int q = 0; q < 4; ++q) {
          SB[r][c + q] = (gr < K && gc + q < N) ? B[gr * N + gc + q] : 0.0f;
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float a_reg[TM], b_reg[TN];
      // A：转置布局让这 TM 个元素连续，两次 float4 读完
#pragma unroll
      for (int m = 0; m < TM; m += 4) {
        const float4 v = *reinterpret_cast<const float4*>(&SA[k][ty * TM + m]);
        a_reg[m + 0] = v.x;
        a_reg[m + 1] = v.y;
        a_reg[m + 2] = v.z;
        a_reg[m + 3] = v.w;
      }
      // B：本身连续
#pragma unroll
      for (int n = 0; n < TN; n += 4) {
        const float4 v = *reinterpret_cast<const float4*>(&SB[k][tx * TN + n]);
        b_reg[n + 0] = v.x;
        b_reg[n + 1] = v.y;
        b_reg[n + 2] = v.z;
        b_reg[n + 3] = v.w;
      }
#pragma unroll
      for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int n = 0; n < TN; ++n) sum[m][n] += a_reg[m] * b_reg[n];
    }
    __syncthreads();
  }

  // ---- 写回 C，也用 float4 ----
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
        for (int q = 0; q < 4; ++q) {
          if (gc + q < N) C[gr * N + gc + q] = sum[m][n + q];
        }
      }
    }
  }
}

void launch_v4_vec4(const float* A, const float* B, float* C, int M, int N, int K) {
  constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
  const dim3 block(BN / TN, BM / TM);  // (16, 16) = 256 线程
  const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_v4_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, N, K);
}

}  // namespace gemm
