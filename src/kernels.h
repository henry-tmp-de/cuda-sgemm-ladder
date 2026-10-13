// kernels.h —— 所有 SGEMM 版本的统一入口声明
//
// 每个版本暴露一个 launch_* 函数，签名一致，方便 main.cu 用一张表驱动。
// 每个 kernel 内部自己决定 grid / block 配置。
#pragma once

namespace gemm {

// v0：naive —— 一个线程算一个 C 元素，直接读全局内存
void launch_v0_naive(const float* A, const float* B, float* C, int M, int N, int K);

// v1：shared memory block tiling（BM=BN=BK=32）
void launch_v1_smem(const float* A, const float* B, float* C, int M, int N, int K);

// v2：1D thread tiling —— 一个线程算同一列的 TM 个元素
void launch_v2_thread1d(const float* A, const float* B, float* C, int M, int N, int K);

// v3：2D thread tiling —— 外积形式，一个线程算 TM x TN
void launch_v3_thread2d(const float* A, const float* B, float* C, int M, int N, int K);

// v4：float4 向量化访存 + A tile 转置存储
void launch_v4_vec4(const float* A, const float* B, float* C, int M, int N, int K);

// v5：手动双缓冲 —— 用寄存器中转把 g2s 搬运和 compute 重叠
void launch_v5_dbuf(const float* A, const float* B, float* C, int M, int N, int K);

// v6：cp.async 三级流水 —— global 直接进 smem，不经过寄存器
void launch_v6_cpasync(const float* A, const float* B, float* C, int M, int N, int K);

}  // namespace gemm
