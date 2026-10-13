#!/bin/bash
# profile_all.sh —— 用 ncu 采集每个 SGEMM 版本的瓶颈证据
#
# 用法（需要 sudo，因为读 GPU 性能计数器需要权限）：
#     sudo bash tools/profile_all.sh [M] [N] [K]
#
# 默认形状 4096x4096x4096。每个版本单独采一次，输出到 profiles/<kernel>.txt。
#
# 选择这些 section 的理由：
#   SpeedOfLight             —— 这个 kernel 是 compute-bound 还是 memory-bound
#   Occupancy                —— 实际占用率，解释「为什么加了这个优化没效果」
#   WarpStateStats           —— stall reason 排名，直接指出瓶颈在哪
#   MemoryWorkloadAnalysis   —— 访存效率、shared memory bank conflict
#   SchedulerStats           —— 指令发射效率
set -u

NCU=/usr/local/cuda/bin/ncu
ROOT="${GEMM_ROOT:-/home/ziru/cuda-gemm-ladder}"
BIN="$ROOT/sgemm_bench"
OUT="$ROOT/profiles"
M="${1:-4096}"
N="${2:-4096}"
K="${3:-4096}"

if [ ! -x "$BIN" ]; then
  echo "!! 找不到 $BIN，请先在 $ROOT 下 make" >&2
  exit 1
fi

mkdir -p "$OUT"

echo "=== ncu 版本 ==="
"$NCU" --version 2>&1 | sed -n '3p'
echo "=== 形状 : ${M}x${N}x${K} ==="
echo "=== 输出 : $OUT ==="
echo

SECTIONS="--section SpeedOfLight --section Occupancy --section WarpStateStats \
--section MemoryWorkloadAnalysis --section SchedulerStats --section LaunchStats"

for v in 0 1 2 3 4 5 6; do
  KERNEL="sgemm_v${v}_kernel"
  echo "--- 采集 $KERNEL ---"
  # --launch-count 1：只 profile 第一次匹配到的 launch，否则 ncu 会把
  # 60 次迭代全部重放一遍，慢几十倍。
  "$NCU" -k "regex:${KERNEL}" --launch-count 1 $SECTIONS \
         "$BIN" "$M" "$N" "$K" > "$OUT/${KERNEL}.txt" 2>&1
  if grep -q "ERR_NV" "$OUT/${KERNEL}.txt"; then
    echo "    !! 失败：$(grep -o 'ERR_NV[A-Z_]*' "$OUT/${KERNEL}.txt" | head -1)"
  else
    echo "    完成 -> $OUT/${KERNEL}.txt"
  fi
done

chmod -R a+r "$OUT" 2>/dev/null
echo
echo "=== 全部完成 ==="
ls -l "$OUT"
