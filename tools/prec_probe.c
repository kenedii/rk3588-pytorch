/*
 * Probe NPU precision codes. B is all ones, so every output is a row sum of A and
 * the (unknown) data layout of a precision cannot affect the result. Rows of A are
 * chosen to reveal range and mantissa width:
 *   row 0: 1.0      row 1: 2^20 (overflows fp16)   row 2: 1 + 2^-8   row 3: 1 + 2^-12
 * For integer inputs the rows are 1, 300, 20000, -12345.
 *
 *   prec_probe IN PROC OUT IN_BYTES OUT_BYTES KGROUP QD BS_OW CVT_SCALE SURF_ADD
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../csrc/rknpu_gemm.h"

static uint16_t bf16(float x) { uint32_t u; memcpy(&u, &x, 4); u += 0x7fff + ((u >> 16) & 1); return u >> 16; }

int main(int argc, char **argv) {
	if (argc < 11) { fprintf(stderr, "usage\n"); return 2; }
	rk_probe_cfg c = { atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]),
	                   atoi(argv[7]), (uint32_t)strtoul(argv[8], 0, 0), (uint32_t)strtoul(argv[9], 0, 0), 0,
	                   (uint32_t)strtoul(argv[10], 0, 0) };
	const int M = 4, K = 32, N = getenv("PN") ? atoi(getenv("PN")) : 16;
	const char *kind = getenv("KIND") ? getenv("KIND") : "f16";   /* f16 bf16 f32 i16 i8 */
	double fv[4] = { 1.0, 1048576.0, 1.0 + 1.0 / 256, 1.0 + 1.0 / 4096 };
	double iv[4] = { 1, 300, 20000, -12345 };
	int isint = kind[0] == 'i';
	static unsigned char A[4 * 32 * 4], W[64 * 32 * 4], C[4 * 64 * 4];
	for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) {
		double v = isint ? iv[m] : fv[m];
		int idx = m * K + k;
		if (!strcmp(kind, "f16")) ((_Float16 *)A)[idx] = (_Float16)v;
		else if (!strcmp(kind, "bf16")) ((uint16_t *)A)[idx] = bf16((float)v);
		else if (!strcmp(kind, "f32")) ((float *)A)[idx] = (float)v;
		else if (!strcmp(kind, "i16")) ((int16_t *)A)[idx] = (int16_t)v;
		else ((int8_t *)A)[idx] = (int8_t)(m == 0 ? 1 : m == 1 ? 100 : m == 2 ? -100 : 7);
	}
	for (int i = 0; i < N * K; i++) {
		if (!strcmp(kind, "f16")) ((_Float16 *)W)[i] = (_Float16)1.0;
		else if (!strcmp(kind, "bf16")) ((uint16_t *)W)[i] = bf16(1.0f);
		else if (!strcmp(kind, "f32")) ((float *)W)[i] = 1.0f;
		else if (!strcmp(kind, "i16")) ((int16_t *)W)[i] = 1;
		else ((int8_t *)W)[i] = 1;
	}
	rk_ctx *ctx = rk_open(NULL);
	if (!ctx) { printf("open failed\n"); return 1; }
	if (rk_probe_raw(ctx, &c, M, N, K, A, W, C)) { printf("RESULT %s in=%d proc=%d out=%d: FAILED (%s)\n", kind, c.in_prec, c.proc_prec, c.out_prec, rk_last_error(ctx)); return 1; }
	printf("RESULT %s in=%d proc=%d out=%d:", kind, c.in_prec, c.proc_prec, c.out_prec);
	for (int m = 0; m < M; m++) {
		double got;
		int idx = m * N + 3;   /* any column: all columns are identical */
		if (c.out_prec == 5) got = ((float *)C)[idx];
		else if (c.out_prec == 4) got = ((int32_t *)C)[idx];
		else if (c.out_prec == 2) got = (double)((_Float16 *)C)[idx];
		else got = ((int16_t *)C)[idx];
		double exp = 32.0 * (isint ? (kind[1] == '8' ? (m == 0 ? 1 : m == 1 ? 100 : m == 2 ? -100 : 7) : iv[m]) : fv[m]);
		printf("  row%d got %.8g (fp64 %.8g)", m, got, exp);
	}
	/* consistency across columns */
	int same = 1;
	for (int m = 0; m < M; m++) for (int n = 1; n < N; n++)
		if (memcmp(C + (m * N) * c.out_bytes, C + (m * N + n) * c.out_bytes, c.out_bytes)) same = 0;
	printf("  cols_equal=%d\n", same);
	rk_close(ctx);
	return 0;
}
