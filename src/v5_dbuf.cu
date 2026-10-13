// v5_dbuf.cu —— 手动双缓冲（把「搬数据」和「算数据」重叠起来）
//
// v4 的执行流是严格串行的：
//     load tile t (g->reg->smem) -> __syncthreads -> compute tile t -> __syncthreads -> load t+1 ...
// 从 global 取数要几百个时钟周期，这段时间计算单元完全闲着。
//
// 双缓冲：shared memory 开两份，算 tile t 的同时把 tile t+1 搬进来。
//     load tile t+1 (global -> reg)   <-- 和下面的 compute 重叠
//     compute tile t (smem)
//     __syncthreads
//     store reg -> smem[另一份]
//     __syncthreads
//
// 中间的「寄存器中转」是关键：先 global->reg 再 reg->smem，中间隔着一次 compute，
// 这样 LDG 发出的长延迟就被 compute 掩盖掉了。这正是 dlog 那篇 v3-dbuf 的做法。
// v6 会用 cp.async 把这个 register 中转整个省掉。
//
// 【踩坑记录】这里刻意**没有用 lambda**。第一版把三段逻辑写成了 [&] 捕获的
// device lambda，编译器为了拿到 ra/rb 的地址把它们放进了 local memory，
// 随后 kernel 抛 "misaligned address"。去掉 lambda、全部内联展开后正常。
// 详见 docs/03-踩坑记录.md。
#include "kernels.h"

namespace gemm {

// 把逻辑写成宏，避免三段代码重复，同时不引入函数/lambda 边界
// （一旦有边界，ra/rb 就可能被塞进 local memory）。
#define V5_LOAD_G2R(T)                                                        \
  do {                                                                        \
    _Pragma("unroll") for (int i = 0; i < A_PER; ++i) {                       \
      const int idx = (tid + i * THREADS) * 4;                                \
      const int r = idx / BK, c = idx % BK;                                   \
      const int gr = a_row_base + r, gc = (T) + c;                            \
      if (gr < M && gc + 3 < K) {                                             \
        ra[i] = *reinterpret_cast<const float4*>(&A[gr * K + gc]);            \
      } else {                                                                \
        float v0 = 0, v1 = 0, v2 = 0, v3 = 0;                                 \
        if (gr < M) {                                                         \
          if (gc + 0 < K) v0 = A[gr * K + gc + 0];                            \
          if (gc + 1 < K) v1 = A[gr * K + gc + 1];                            \
          if (gc + 2 < K) v2 = A[gr * K + gc + 2];                            \
          if (gc + 3 < K) v3 = A[gr * K + gc + 3];                            \
        }                                                                     \
        ra[i] = make_float4(v0, v1, v2, v3);                                  \
      }                                                                       \
    }                                                                         \
    _Pragma("unroll") for (int i = 0; i < B_PER; ++i) {                       \
      const int idx = (tid + i * THREADS) * 4;                                \
      const int r = idx / BN, c = idx % BN;                                   \
      const int gr = (T) + r, gc = b_col_base + c;                            \
      if (gr < K && gc + 3 < N) {                                             \
        rb[i] = *reinterpret_cast<const float4*>(&B[gr * N + gc]);            \
      } else {                                                                \
        float v0 = 0, v1 = 0, v2 = 0, v3 = 0;                                 \
        if (gr < K) {                                                         \
          if (gc + 0 < N) v0 = B[gr * N + gc + 0];                            \
          if (gc + 1 < N) v1 = B[gr * N + gc + 1];                            \
          if (gc + 2 < N) v2 = B[gr * N + gc + 2];                            \
          if (gc + 3 < N) v3 = B[gr * N + gc + 3];                            \
        }                                                                     \
        rb[i] = make_float4(v0, v1, v2, v3);                                  \
      }                                                                       \
    }                                                                         \
  } while (0)

#define V5_STORE_R2S(S)                                                       \
  do {                                                                        \
    _Pragma("unroll") for (int i = 0; i < A_PER; ++i) {                       \
      const int idx = (tid + i * THREADS) * 4;                                \
      const int r = idx / BK, c = idx % BK;                                   \
      SA[(S)][c + 0][r] = ra[i].x; /* 转置：按列写 */                          \
      SA[(S)][c + 1][r] = ra[i].y;                                            \
      SA[(S)][c + 2][r] = ra[i].z;                                            \
      SA[(S)][c + 3][r] = ra[i].w;                                            \
    }                                                                         \
    _Pragma("unroll") for (int i = 0; i < B_PER; ++i) {                       \
      const int idx = (tid + i * THREADS) * 4;                                \
      const int r = idx / BN, c = idx % BN;                                   \
      *reinterpret_cast<float4*>(&SB[(S)][r][c]) = rb[i];                     \
    }                                                                         \
  } while (0)

#define V5_COMPUTE(S)                                                         \
  do {                                                                        \
    _Pragma("unroll") for (int k = 0; k < BK; ++k) {                          \
      float a_reg[TM], b_reg[TN];                                             \
      _Pragma("unroll") for (int m = 0; m < TM; m += 4) {                     \
        const float4 v =                                                      \
            *reinterpret_cast<const float4*>(&SA[(S)][k][ty * TM + m]);       \
        a_reg[m + 0] = v.x; a_reg[m + 1] = v.y;                               \
        a_reg[m + 2] = v.z; a_reg[m + 3] = v.w;                               \
      }                                                                       \
      _Pragma("unroll") for (int n = 0; n < TN; n += 4) {                     \
        const float4 v =                                                      \
            *reinterpret_cast<const float4*>(&SB[(S)][k][tx * TN + n]);       \
        b_reg[n + 0] = v.x; b_reg[n + 1] = v.y;                               \
        b_reg[n + 2] = v.z; b_reg[n + 3] = v.w;                               \
      }                                                                       \
      _Pragma("unroll") for (int m = 0; m < TM; ++m)                          \
          _Pragma("unroll") for (int n = 0; n < TN; ++n)                      \
              sum[m][n] += a_reg[m] * b_reg[n];                               \
    }                                                                         \
  } while (0)

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__(((BM / TM) * (BN / TN)))
sgemm_v5_kernel(const float* __restrict__ A,
                const float* __restrict__ B,
                float* __restrict__ C,
                int M,
                int N,
                int K) {
  static_assert(BK % 4 == 0 && TM % 4 == 0 && TN % 4 == 0, "need float4 alignment");
  constexpr int THREADS = (BM / TM) * (BN / TN);
  constexpr int A_PER = BM * BK / 4 / THREADS;
  constexpr int B_PER = BK * BN / 4 / THREADS;
  static_assert((BM * BK / 4) % THREADS == 0, "A tile/threads mismatch");
  static_assert((BK * BN / 4) % THREADS == 0, "B tile/threads mismatch");

  // 双缓冲：两份 smem。__align__(16) 保证 float4 读写合法。
  __shared__ __align__(16) float SA[2][BK][BM];  // A 转置存放
  __shared__ __align__(16) float SB[2][BK][BN];

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

  float4 ra[A_PER], rb[B_PER];  // 寄存器中转缓冲

  const int num_tiles = (K + BK - 1) / BK;

  V5_LOAD_G2R(0);
  V5_STORE_R2S(0);
  __syncthreads();

  for (int t = 0; t < num_tiles; ++t) {
    const int cur = t & 1;
    if (t + 1 < num_tiles) {
      // 注意宏的参数是【K 方向的字节偏移】，必须是 BK 的倍数，不是 tile 序号。
      // 这里踩过一个坑：写成 V5_LOAD_G2R(t + 1) 会让 gc 整体错位 1 个 float，
      // kernel 抛 "misaligned address"。详见 docs/03-踩坑记录.md。
      V5_LOAD_G2R((t + 1) * BK);  // 发出全局加载，延迟被下面的 compute 掩盖
    }
    V5_COMPUTE(cur);
    if (t + 1 < num_tiles) {
      __syncthreads();          // 确认所有线程算完 cur
      V5_STORE_R2S(cur ^ 1);    // 写入另一层
      __syncthreads();          // 确认写完，下一轮可读
    }
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

#undef V5_LOAD_G2R
#undef V5_STORE_R2S
#undef V5_COMPUTE

void launch_v5_dbuf(const float* A, const float* B, float* C, int M, int N, int K) {
  constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
  const dim3 block(BN / TN, BM / TM);  // (16, 16) = 256 线程
  const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_v5_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, N, K);
}

}  // namespace gemm
