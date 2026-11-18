/* Correctness + timing check for rk_gemm_f32 against a CPU reference. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../csrc/rknpu_gemm.h"

static float randn(void) {
	float u = (rand() + 1.0f) / (RAND_MAX + 2.0f), v = (rand() + 1.0f) / (RAND_MAX + 2.0f);
	return sqrtf(-2.f * logf(u)) * cosf(6.2831853f * v);
}

int main(int argc, char **argv) {
	if (argc < 4) { fprintf(stderr, "usage: %s M N K [reps] [transA] [transB] [check]\n", argv[0]); return 2; }
	int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
	int reps = argc > 4 ? atoi(argv[4]) : 3;
	int ta = argc > 5 ? atoi(argv[5]) : 0, tb = argc > 6 ? atoi(argv[6]) : 0;
	int check = argc > 7 ? atoi(argv[7]) : 1;
	rk_ctx *ctx = rk_open(NULL);
	if (!ctx) { fprintf(stderr, "cannot open rknpu device\n"); return 1; }
	float *A = malloc(sizeof(float) * M * K), *B = malloc(sizeof(float) * K * N);
	float *C = malloc(sizeof(float) * M * N);
	srand(1234);
	for (long i = 0; i < (long)M * K; i++) A[i] = randn();
	for (long i = 0; i < (long)K * N; i++) B[i] = randn();
	/* logical A is MxK, B is KxN; storage optionally transposed */
	long sam = ta ? 1 : K, sak = ta ? M : 1;
	long sbk = tb ? 1 : N, sbn = tb ? K : 1;
	rk_stats st; double best = 1e30;
	for (int r = 0; r < reps; r++) {
		memset(C, 0, sizeof(float) * M * N);
		if (rk_gemm_f32(ctx, M, N, K, A, sam, sak, B, sbk, sbn, C, N, 0)) {
			fprintf(stderr, "rep %d: gemm failed: %s\n", r, rk_last_error(ctx));
			if (!getenv("RK_KEEP_GOING")) return 1;
			continue;
		}
		if (getenv("RK_EACH")) {
			int bad = 0;
			for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
				double ref = 0;
				for (int k = 0; k < K; k++) ref += (double)(_Float16)A[m * sam + k * sak] * (double)(_Float16)B[k * sbk + n * sbn];
				if (fabs(ref - C[(long)m * N + n]) > 1e-2 * sqrt((double)K) + 1e-3 * fabs(ref)) bad++;
			}
			printf("rep %d: ok, bad=%d/%d\n", r, bad, M * N);
		}
		rk_get_stats(ctx, &st);
		if (st.total_us < best) best = st.total_us;
	}
	double gflop = 2.0 * M * N * K / 1e9;
	printf("M=%d N=%d K=%d tA=%d tB=%d | tiles m%d k%d n%d tasks %d submits %d | pack %.0fus sync %.0fus submit %.0fus hw %.0fus unpack %.0fus total %.0fus | %.1f GFLOPS (best %.1f)\n",
	       M, N, K, ta, tb, st.m_tile, st.k_tile, st.n_tile, st.tasks, st.submits, st.pack_us, st.sync_us, st.submit_us, st.hw_us,
	       st.unpack_us, st.total_us, gflop / (st.total_us / 1e6), gflop / (best / 1e6));
	if (check) {
		/* reference uses fp16-rounded inputs so the residual is accumulation/output error only */
		double maxerr = 0, maxref = 0, sse = 0, ssr = 0; int bad = 0;
		int step_m = M > 256 ? M / 128 : 1;
		for (int m = 0; m < M; m += step_m) for (int n = 0; n < N; n++) {
			double ref = 0;
			for (int k = 0; k < K; k++) ref += (double)(_Float16)A[m * sam + k * sak] * (double)(_Float16)B[k * sbk + n * sbn];
			double e = fabs(ref - C[(long)m * N + n]);
			if (e > maxerr) maxerr = e;
			if (fabs(ref) > maxref) maxref = fabs(ref);
			sse += e * e; ssr += ref * ref;
			if (e > 1e-2 * sqrt((double)K) + 1e-3 * fabs(ref)) { if (bad < 3) printf("  bad m=%d n=%d ref=%f got=%f\n", m, n, ref, C[(long)m * N + n]); bad++; }
		}
		if (getenv("RK_MAP") && M <= 128 && N <= 128) {
			for (int m = 0; m < M; m++) {
				char line[129];
				for (int n = 0; n < N; n++) {
					double ref = 0;
					for (int k = 0; k < K; k++) ref += (double)(_Float16)A[m * sam + k * sak] * (double)(_Float16)B[k * sbk + n * sbn];
					double got = C[(long)m * N + n];
					line[n] = fabs(ref - got) <= 1e-2 * sqrt((double)K) + 1e-3 * fabs(ref) ? '.' : (got == 0 ? '0' : 'X');
				}
				line[N] = 0;
				printf("  %3d %s\n", m, line);
			}
		}
		printf("  check: max_abs_err %.3e  rel_rms_err %.3e  max|ref| %.1f  bad %d  -> %s\n", maxerr, sqrt(sse / (ssr + 1e-30)), maxref, bad, bad ? "FAIL" : "PASS");
	}
	rk_close(ctx);
	return 0;
}
