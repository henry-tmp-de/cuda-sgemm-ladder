# 04 · 时间线分析（nsys → Perfetto）

## 先说清楚：ncu 和 Perfetto 不是一回事

|  | ncu（`docs/02` 用的） | nsys + Perfetto（这篇） |
|---|---|---|
| 视角 | **单个 kernel 内部** | **整条时间线** |
| 数据 | 硬件性能计数器 | kernel/memcpy 的起止时间与顺序 |
| 回答 | "这个 kernel 为什么慢"——L1 吞吐、bank conflict、stall | "谁在什么时候跑、有没有空档、顺序对不对" |
| 产物 | `.ncu-rep` | `.nsys-rep` → Chrome Trace JSON |

**两者互补**：先用 nsys/Perfetto 看**哪一段值得深挖**，再用 ncu 钻进去看**为什么**。
这也是 LeetCUDA 第 1 章定的顺序——"先 nsys，后 ncu"。

---

## 怎么采、怎么转

```bash
# 1) 编译时打开 NVTX 区间标记（否则时间线上只有一堆分不清的 kernel 名字）
make clean && make NVTX=1

# 2) 采时间线（不需要管理员权限，nsys 不读性能计数器）
nsys profile -o gemm_trace --stats=false ./sgemm_bench 1024 1024 1024

# 3) 导出为 SQLite
nsys export --type=sqlite --force-overwrite=true -o gemm_trace.sqlite gemm_trace.nsys-rep

# 4) 转成 Chrome Trace Event Format
python3 tools/nsys2perfetto.py gemm_trace.sqlite perfetto/gemm_trace.json

# 5) 打开 https://ui.perfetto.dev ，把 gemm_trace.json 拖进去
```

### 为什么要自己写转换（`tools/nsys2perfetto.py`）

`nsys export --type=json` 产出的是 **nsys 自己的 schema**（顶层是 `{"data":[...]}`），
**Perfetto / ui.perfetto.dev 读不了**。Perfetto 认的是 Chrome Trace Event Format：

```json
[{"name":"...", "cat":"kernel", "ph":"X", "ts":<微秒>, "dur":<微秒>, "pid":0, "tid":<流号>, "args":{...}}]
```

两个关键点：
1. **CUPTI 的时间戳单位是纳秒，Chrome Trace 是微秒**，要除以 1000
2. `tid` 用 **CUDA stream 号**，这样每个流在 Perfetto 上是一条独立泳道

转换器还把 `grid` / `block` / `registersPerThread` / 共享内存大小塞进了事件的 `args`，
点开 Perfetto 里任意一个 kernel 就能看到它的发射配置。

---

## 从时间线上看到了什么

### 1. 每一级优化都肉眼可见地变短了

时间线上一段一段排下来（每段是一个 NVTX 区间），**段宽就是这一版跑 60 次的总时间**：

| NVTX 分段 | 段内墙钟 |
|---|---|
| cuBLAS(FP32) | 129.7 ms |
| v0-naive | 90.1 ms |
| v1-smem | 77.0 ms |
| v2-thread1d | 53.7 ms |
| v3-thread2d | 40.7 ms |
| v4-vec4 | 38.8 ms |
| v5-dbuf | 34.6 ms |
| v6-cpasync | 33.1 ms |
| v7-lbcap | 33.6 ms |

（1024³。注意 `v7` 比 `v6` 略慢，和 `docs/02` 里的结论一致。）

### 2. 意外发现：L2 冲刷 kernel 占了 162 ms

按 kernel 的 GPU 总耗时排序：

```
gemm::l2_flush_kernel      162.09 ms   ← 比所有 GEMM 加起来还多
gemm::sgemm_v0_kernel       62.45 ms
gemm::sgemm_v1_kernel       49.80 ms
...
```

**这是我测量框架自己的开销。** 为了让 benchmark 测到"冷缓存"的真实数字，
每轮迭代前都要读一遍 256 MB 的缓冲把 L2 挤出去
（见 `docs/01-测量方法.md`），代价是这些读操作本身也占 GPU 时间。

**它不影响测出来的 GEMM 数字**——冲刷在计时区间之外，且和被测 kernel 在同一个流上串行。
但它是个真实的取舍：**为了让单次测量准确，整个 benchmark 的墙钟时间被拉长了一倍多。**

> 如果只是跑一次看结果，这没关系；但如果要跑大规模参数扫描，这个开销就值得优化
> （比如缩小冲刷缓冲、或者只在测第一个版本前冲一次）。

### 3. cuBLAS 的 kernel 配置被时间线暴露了

时间线里 cuBLAS 段用的是它内部的 kernel，名字和配置都能看到：

```
kernel : ampere_sgemm_128x64_nn
grid   : (8, 16, 5)
block  : (128, 1, 1)
regs   : 122 / thread
smem   : 12544 B
```

**可以直接读出 cuBLAS 在 1024³ 上的策略**：128×64 的 block tile（我手写的是 128×128）、
每线程 122 个寄存器（我 v4 是 125，非常接近）、grid 的第三维是 5。

> 这是时间线的额外价值：**它让你能"偷看"商业库的实现选择**，
> 拿它和自己的配置对照。面试时能说出"cuBLAS 在这个尺寸用的是 128×64 tile，
> 用的是 122 个寄存器"是很加分的细节。

---

### 4. CPU 轨道：可以看出发射开销和库的内部开销

补上 `CUPTI_ACTIVITY_KIND_RUNTIME` 之后，时间线上多出一条 **CPU (CUDA API)** 轨道，
显示每个 `cudaXxx` 调用在 CPU 上花了多久。统计下来：

```
cudaLaunchKernel_v7000                    999
cudaEventRecord_v3020                     961
cuGetProcAddress_v2                       838   ← 不是我们写的
cudaEventSynchronize_v3020                450
cudaStreamGetCaptureInfo_v2_v11030        305   ← 也不是我们写的
cudaMemsetAsync_v3020                      61
```

**注意 `cuGetProcAddress` 838 次和 `cudaStreamGetCaptureInfo` 305 次** ——
这两类调用**在我们的源码里一次都没出现过**，是 CUDA runtime 的内部开销
（符号解析、Graph capture 状态查询）。

> 这是一个用时间线才能发现的东西：**你的程序在 CPU 侧发起的调用，比你写的行数多得多。**
> 对小 kernel 来说，这些固定开销会变成瓶颈——这也是为什么要做 kernel fusion、
> 为什么要用 CUDA Graph 把一串发射"录"下来重放。
> 本项目每个 kernel 都有毫秒级，这点开销可以忽略；但如果换成 elementwise 小算子，就不能忽视了。

### 5. 轨道结构总览

导出的 trace 一共三条 process、四个泳道：

| pid | 轨道 | tid | 内容 |
|---|---|---|---|
| 0 | **GPU (CUDA)** | 7 | 1087 个事件：kernel / memset / memcpy |
| 0 | GPU (CUDA) | 4294967295 | 469 个**设备级同步**（`0xFFFFFFFF` 是"无流"哨兵值，单独一条泳道） |
| 1 | **NVTX 区间** | 0 | 9 个版本分段 |
| 2 | **CPU (CUDA API)** | 0 | 3747 个 `cudaXxx` 调用（线程名 `sgemm_bench`） |

---

## 一个使用纪律（和 ncu 一样）

**时间线上的时长不能当作性能数字。** nsys 挂上 CUPTI 插桩后，kernel 的执行会被拖慢，
上表里 v6 段内 60 次迭代的总墙钟（33.1 ms）和 `./sgemm_bench` 直接测出来的
（0.127 ms × 60 ≈ 7.6 ms）差了 4 倍多。

| 用途 | 用哪个数字 |
|---|---|
| 看**结构**：谁先谁后、有没有空隙、有几个 kernel | ✅ 时间线 |
| 看**比例**：哪个 kernel 占大头 | ✅ 时间线 |
| 看**绝对性能** | ❌ 用 `./sgemm_bench` |

这和 `docs/03-踩坑记录.md` 里那条"ncu 的 Duration 不是性能数字"是同一个道理：
**附加工具会改变被测量的对象。**
