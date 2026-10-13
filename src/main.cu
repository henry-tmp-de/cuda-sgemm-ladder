// main.cu —— benchmark 驱动
//
// 用法：./sgemm_bench [M] [N] [K]
// 默认 4096 4096 4096
//
// 输出一张表：每个版本的中位耗时、GFLOPS、相对 cuBLAS 的百分比、最大误差。
#include "bench.cuh"
#include "kernels.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Entry {
  const char* name;
  gemm::LaunchFn fn;
  const char* note;
};

}  // namespace

int main(int argc, char** argv) {
  int M = 4096, N = 4096, K = 4096;
  if (argc >= 4) {
    M = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
    K = std::atoi(argv[3]);
  } else if (argc >= 2 && std::strcmp(argv[1], "--quick") == 0) {
    M = N = K = 1024;
  }

  // ---- 打印运行环境，保证数字可复现 ----
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("=== SGEMM Ladder Benchmark ===\n");
  std::printf("GPU        : %s (sm_%d%d, %d SMs)\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount);
  std::printf("Shape      : M=N=K=%d  (%.2f GFLOP)\n", M,
              2.0 * M * N * K / 1e9);
  // clockRate 单位是 kHz -> 乘 1e3 得 Hz -> 除 1e12 得 TFLOPS，合起来 1e-9
  std::printf("Peak FP32  : %.2f TFLOPS (%.0f MHz x %d cores x 2)\n",
              prop.multiProcessorCount * 128.0 * prop.clockRate * 1e-9 * 2.0,
              prop.clockRate * 1e-3, prop.multiProcessorCount * 128);
  std::printf("Mem BW     : %.1f GB/s (theoretical)\n",
              prop.memoryClockRate * 1e-6 * (prop.memoryBusWidth / 8.0) * 2.0);
  std::printf("cuBLAS     : CUBLAS_DEFAULT_MATH (strict FP32, no TF32)\n\n");

  // ---- 准备数据 ----
  const size_t bytes_a = static_cast<size_t>(M) * K * sizeof(float);
  const size_t bytes_b = static_cast<size_t>(K) * N * sizeof(float);
  const size_t bytes_c = static_cast<size_t>(M) * N * sizeof(float);

  float *A, *B, *C, *C_ref;
  CUDA_CHECK(cudaMalloc(&A, bytes_a));
  CUDA_CHECK(cudaMalloc(&B, bytes_b));
  CUDA_CHECK(cudaMalloc(&C, bytes_c));
  CUDA_CHECK(cudaMalloc(&C_ref, bytes_c));

  std::vector<float> h_a(M * K), h_b(K * N);
  srand(42);
  for (auto& v : h_a) v = static_cast<float>(rand()) / RAND_MAX - 0.5f;
  for (auto& v : h_b) v = static_cast<float>(rand()) / RAND_MAX - 0.5f;
  CUDA_CHECK(cudaMemcpy(A, h_a.data(), bytes_a, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B, h_b.data(), bytes_b, cudaMemcpyHostToDevice));

  gemm::L2Flusher flusher;

  // ---- cuBLAS 基线 ----
  gemm::cublas_sgemm(A, B, C_ref, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());
  gemm::Timing t_ref = gemm::time_cublas(A, B, C_ref, M, N, K, flusher);
  const double cublas_gf = gemm::gflops(M, N, K, t_ref.median_ms);

  // ---- 各版本 ----
  const std::vector<Entry> entries = {
      {"v0-naive", gemm::launch_v0_naive, "one thread per element"},
      {"v1-smem", gemm::launch_v1_smem, "32x32 block tile"},
      {"v2-thread1d", gemm::launch_v2_thread1d, "TM=8, one column per thread"},
      {"v3-thread2d", gemm::launch_v3_thread2d, "TM=TN=8, outer product"},
      {"v4-vec4", gemm::launch_v4_vec4, "float4 + transposed A tile"},
      {"v5-dbuf", gemm::launch_v5_dbuf, "manual double buffering"},
      {"v6-cpasync", gemm::launch_v6_cpasync, "cp.async 3-stage pipeline"},
      {"v7-lbcap", gemm::launch_v7_lbcap, "v6 + launch_bounds(256,2)"},
  };

  std::printf("%-14s %12s %12s %10s %12s  %s\n", "version", "median(ms)",
              "GFLOPS", "vs cuBLAS", "max_err", "note");
  std::printf("%-14s %12.4f %12.1f %9.1f%% %12s  %s\n", "cuBLAS(FP32)",
              t_ref.median_ms, cublas_gf, 100.0, "-", "baseline");

  std::vector<float> h_c(M * N);
  for (const auto& e : entries) {
    CUDA_CHECK(cudaMemset(C, 0, bytes_c));
    e.fn(A, B, C, M, N, K);
    // 关键：launch 之后立刻查错。只靠 cudaDeviceSynchronize 有时会漏掉
    // “invalid configuration argument” 这类配置错误，表现为 kernel 根本没跑、
    // 耗时接近 0、结果全是 0 —— 却又不报错。
    {
      cudaError_t lerr = cudaGetLastError();
      if (lerr != cudaSuccess) {
        std::printf("%-14s   !! LAUNCH FAILED: %s\n", e.name,
                    cudaGetErrorString(lerr));
        continue;
      }
    }
    // 运行期错误也不中断整个 benchmark：报出来然后跳到下一个版本，
    // 否则一个版本崩了后面全看不到。
    {
      cudaError_t serr = cudaDeviceSynchronize();
      if (serr != cudaSuccess) {
        std::printf("%-14s   !! RUNTIME ERROR: %s\n", e.name,
                    cudaGetErrorString(serr));
        (void)cudaGetLastError();  // 清掉 sticky error
        continue;
      }
    }

    CUDA_CHECK(cudaMemcpy(h_c.data(), C, bytes_c, cudaMemcpyDeviceToHost));
    std::vector<float> h_ref(M * N);
    CUDA_CHECK(cudaMemcpy(h_ref.data(), C_ref, bytes_c, cudaMemcpyDeviceToHost));
    const double err = gemm::max_abs_diff(h_c.data(), h_ref.data(), M * N);
    // 诊断：把两者的量级也打出来。如果只报“误差=0”而不知道量级，
    // 万一是“两边都是 0”的假一致就发现不了。
    double c_max = 0.0, ref_max = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(M) * N; ++i) {
      c_max = std::max(c_max, std::fabs(static_cast<double>(h_c[i])));
      ref_max = std::max(ref_max, std::fabs(static_cast<double>(h_ref[i])));
    }
    std::printf("               [diag] max|C|=%.3f  max|C_ref|=%.3f\n", c_max, ref_max);

    gemm::Timing t = gemm::time_launch(e.fn, A, B, C, M, N, K, flusher);
    const double gf = gemm::gflops(M, N, K, t.median_ms);
    std::printf("%-14s %12.4f %12.1f %9.1f%% %12.2e  %s\n", e.name,
                t.median_ms, gf, gf / cublas_gf * 100.0, err, e.note);
  }

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C_ref);
  return 0;
}
