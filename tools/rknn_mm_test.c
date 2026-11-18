/* fp16 matmul through Rockchip's rknn_matmul API (librknnrt), used as a reference backend. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "rknn_matmul_api.h"

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }

int main(int argc, char **argv) {
	int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
	int reps = argc > 4 ? atoi(argv[4]) : 3;
	int core = argc > 5 ? atoi(argv[5]) : 0;
	rknn_matmul_ctx ctx = 0;
	rknn_matmul_info info; memset(&info, 0, sizeof(info));
	info.M = M; info.K = K; info.N = N;
	info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
	info.B_layout = 0; info.AC_layout = 0;
	rknn_matmul_io_attr io; memset(&io, 0, sizeof(io));
	int r = rknn_matmul_create(&ctx, &info, &io);
	if (r) { printf("rknn_matmul_create failed %d\n", r); return 1; }
	printf("A size %u B size %u C size %u\n", io.A.size, io.B.size, io.C.size);
	if (core) rknn_matmul_set_core_mask(ctx, (rknn_core_mask)core);
	rknn_tensor_mem *a = rknn_create_mem(ctx, io.A.size), *b = rknn_create_mem(ctx, io.B.size), *c = rknn_create_mem(ctx, io.C.size);
	_Float16 *pa = a->virt_addr, *pb = b->virt_addr; float *pc = c->virt_addr;
	srand(1);
	for (int i = 0; i < M * K; i++) pa[i] = (_Float16)((rand() % 200 - 100) / 50.0f);
	for (int i = 0; i < K * N; i++) pb[i] = (_Float16)((rand() % 200 - 100) / 50.0f);
	if (getenv("B_INDEX")) for (int k = 0; k < K; k++) for (int n = 0; n < N; n++) pb[k * N + n] = (_Float16)(k + K * n);
	rknn_matmul_set_io_mem(ctx, a, &io.A); rknn_matmul_set_io_mem(ctx, b, &io.B); rknn_matmul_set_io_mem(ctx, c, &io.C);
	double best = 1e30;
	for (int i = 0; i < reps; i++) {
		double t0 = now_us();
		r = rknn_matmul_run(ctx);
		double dt = now_us() - t0;
		if (dt < best) best = dt;
		if (r) { printf("run failed %d\n", r); return 1; }
	}
	double maxerr = 0;
	for (int m = 0; m < M; m += (M > 64 ? M / 64 : 1)) for (int n = 0; n < N; n++) {
		double ref = 0; for (int k = 0; k < K; k++) ref += (double)pa[m * K + k] * (double)pb[k * N + n];
		double e = fabs(ref - pc[m * N + n]); if (e > maxerr) maxerr = e;
	}
	printf("rknn_matmul M=%d K=%d N=%d core=%d best %.0f us  %.1f GFLOPS  max_abs_err %.3e\n", M, K, N, core, best, 2.0 * M * N * K / best / 1e3, maxerr);
	rknn_destroy_mem(ctx, a); rknn_destroy_mem(ctx, b); rknn_destroy_mem(ctx, c);
	rknn_matmul_destroy(ctx);
	return 0;
}
