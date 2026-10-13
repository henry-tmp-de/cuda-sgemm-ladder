# CUDA SGEMM 优化阶梯
#
# 用法：
#   make              # 编译
#   make run          # 4096^3
#   make quick        # 1024^3，快速迭代
#   make clean

ARCH    ?= sm_86
CUDA    ?= /usr/local/cuda
NVCC    := $(CUDA)/bin/nvcc
TARGET  := sgemm_bench

# 不加 --use_fast_math：保证 FP32 语义与 cuBLAS 严格一致，
# 否则“达到 cuBLAS 的百分之几”就不是同精度对比了。
NVCCFLAGS := -O3 -arch=$(ARCH) -std=c++17 -lineinfo \
             --generate-line-info \
             -Xcompiler -Wall

# NVTX=1 时打开时间线标记，配合 nsys 采集后在 ui.perfetto.dev 上查看：
#   make clean && make NVTX=1
#   nsys profile -o gemm_trace ./sgemm_bench 1024 1024 1024
#   nsys export --type=json --force-overwrite=true -o gemm_trace.json gemm_trace.nsys-rep
ifeq ($(NVTX),1)
NVCCFLAGS += -DUSE_NVTX
endif

SRC := $(wildcard src/*.cu)
OBJ := $(SRC:.cu=.o)

$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lcublas

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

run: $(TARGET)
	./$(TARGET) 4096 4096 4096

quick: $(TARGET)
	./$(TARGET) --quick

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: run quick clean
