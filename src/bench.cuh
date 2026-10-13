// bench.cuh —— 所有 kernel 版本共用的测量与校验工具
//
// 这个文件是整个项目的“尺子”。测量口径错了，后面所有数字都是假的，
// 所以这里刻意把三件容易被做错的事写死：
//   1) 计时用 CUDA Event，不用 CPU 时钟
//   2) 每轮之间冲掉 L2，避免第二次跑数据还在缓存里
//   3) 多次测量取中位数，且先 warmup 让 GPU 升频
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// 错误检查
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t _e = (expr);                                                  \
    if (_e != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error: %s  (%s:%d)\n",                       \
                   cudaGetErrorString(_e), __FILE__, __LINE__);               \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

#define CUBLAS_CHECK(expr)                                                    \
  do {                                                                        \
    cublasStatus_t _s = (expr);                                               \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                        \
      std::fprintf(stderr, "cuBLAS error: %d  (%s:%d)\n", (int)_s, __FILE__,  \
                   __LINE__);                                                 \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

namespace gemm {

// ---------------------------------------------------------------------------
// 计算量。C = A(MxK) * B(KxN)，每个输出元素做 K 次乘加 = 2K FLOP
// ---------------------------------------------------------------------------
inline double gflops(int M, int N, int K, double ms) {
  // 2 * M * N * K 次浮点运算 / 毫秒 -> GFLOPS
  return 2.0 * static_cast<double>(M) * N * K / (ms * 1e-3) / 1e9;
}

// ---------------------------------------------------------------------------
// L2 冲刷。3090 的 L2 有 6MB，这里开 256MB 的缓冲，每轮读一遍把它挤出去。
// 不这样做的话，第二次以后的迭代会命中 L2，测出来的是“假快”。
// ---------------------------------------------------------------------------
__global__ void l2_flush_kernel(const float* __restrict__ p,
                                size_t n,
                                float* __restrict__ sink) {
  float acc = 0.0f;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    acc += p[i];
  }
  // 这个条件永远为假，作用是让编译器不能把上面的循环优化掉
  if (acc == 12345.678f) *sink = acc;
}

class L2Flusher {
 public:
  explicit L2Flusher(size_t bytes = 256ull << 20) {
    n_ = bytes / sizeof(float);
    CUDA_CHECK(cudaMalloc(&buf_, n_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(buf_, 0, n_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sink_, sizeof(float)));
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount,
                                      dev));
  }
  ~L2Flusher() {
    cudaFree(buf_);
    cudaFree(sink_);
  }
  L2Flusher(const L2Flusher&) = delete;
  L2Flusher& operator=(const L2Flusher&) = delete;

  void flush(cudaStream_t stream = nullptr) const {
    l2_flush_kernel<<<sm_count_ * 8, 256, 0, stream>>>(buf_, n_, sink_);
  }

 private:
  float* buf_ = nullptr;
  float* sink_ = nullptr;
  size_t n_ = 0;
  int sm_count_ = 0;
};

// ---------------------------------------------------------------------------
// 计时。warmup 次预热（顺带让 GPU 把频率升上去），然后 repeat 次计时取中位数。
// ---------------------------------------------------------------------------
using LaunchFn = void (*)(const float*, const float*, float*, int, int, int);

struct Timing {
  double median_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
};

inline Timing time_launch(LaunchFn launch,
                          const float* A,
                          const float* B,
                          float* C,
                          int M,
                          int N,
                          int K,
                          const L2Flusher& flusher,
                          int warmup = 10,
                          int repeat = 50) {
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  for (int i = 0; i < warmup; ++i) {
    launch(A, B, C, M, N, K);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> samples;
  samples.reserve(repeat);
  for (int i = 0; i < repeat; ++i) {
    flusher.flush();  // 每轮先把 L2 冲掉，保证测的是冷缓存下的真实访存
    CUDA_CHECK(cudaEventRecord(ev_start));
    launch(A, B, C, M, N, K);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    samples.push_back(static_cast<double>(ms));
  }

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));

  std::sort(samples.begin(), samples.end());
  Timing t;
  t.median_ms = samples[samples.size() / 2];
  t.min_ms = samples.front();
  t.max_ms = samples.back();
  return t;
}

// ---------------------------------------------------------------------------
// cuBLAS 基线。
//
// 注意 CUBLAS_DEFAULT_MATH：这会显式关掉 TF32，保证对照组是**严格 FP32**，
// 和我们的手写 kernel 同精度。不设的话在某些路径上 cuBLAS 会走 TF32，
// 那样“达到 cuBLAS 的百分之几”就不是同精度对比了。
// ---------------------------------------------------------------------------
inline void cublas_sgemm(const float* A,
                         const float* B,
                         float* C,
                         int M,
                         int N,
                         int K) {
  static cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  }
  const float alpha = 1.0f, beta = 0.0f;
  // cuBLAS 是列主序，所以要用 C^T = B^T * A^T 的等价写法
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                           B, N, A, K, &beta, C, N));
}

inline Timing time_cublas(const float* A,
                          const float* B,
                          float* C,
                          int M,
                          int N,
                          int K,
                          const L2Flusher& flusher,
                          int warmup = 10,
                          int repeat = 50) {
  return time_launch(&cublas_sgemm, A, B, C, M, N, K, flusher, warmup, repeat);
}

// ---------------------------------------------------------------------------
// 正确性校验：和 cuBLAS 的结果逐元素比最大绝对误差
// ---------------------------------------------------------------------------
inline double max_abs_diff(const float* x, const float* y, size_t n) {
  double worst = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double d = std::fabs(static_cast<double>(x[i]) - static_cast<double>(y[i]));
    if (d > worst) worst = d;
  }
  return worst;
}

}  // namespace gemm
