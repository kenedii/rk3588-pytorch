CC ?= gcc
CFLAGS ?= -O3 -march=armv8.2-a+fp16 -fPIC -pthread -Wall -Wno-unused-function
all: librknpu_gemm.so gemm_test prec_probe
librknpu_gemm.so: rknpu_gemm.c rknpu_eltwise.c rknpu_dwconv.c rknpu_bnorm.c rknpu_gemm.h rknpu_uapi.h npu_regs.h
	$(CC) $(CFLAGS) -shared -o $@ rknpu_gemm.c
gemm_test: ../tools/gemm_test.c librknpu_gemm.so
	$(CC) $(CFLAGS) -o $@ ../tools/gemm_test.c -L. -lrknpu_gemm -Wl,-rpath,'$$ORIGIN' -lm
prec_probe: ../tools/prec_probe.c librknpu_gemm.so
	$(CC) $(CFLAGS) -o $@ ../tools/prec_probe.c -L. -lrknpu_gemm -Wl,-rpath,'$$ORIGIN' -lm
clean:
	rm -f librknpu_gemm.so gemm_test prec_probe
.PHONY: all clean
