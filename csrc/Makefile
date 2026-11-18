CC ?= gcc
CFLAGS ?= -O3 -march=armv8.2-a+fp16 -fopenmp -fPIC -Wall -Wno-unused-function
all: librknpu_gemm.so gemm_test
librknpu_gemm.so: rknpu_gemm.c rknpu_gemm.h rknpu_uapi.h npu_regs.h
	$(CC) $(CFLAGS) -shared -o $@ rknpu_gemm.c
gemm_test: ../tools/gemm_test.c librknpu_gemm.so
	$(CC) $(CFLAGS) -o $@ ../tools/gemm_test.c -L. -lrknpu_gemm -Wl,-rpath,'$$ORIGIN' -lm
clean:
	rm -f librknpu_gemm.so gemm_test
.PHONY: all clean
