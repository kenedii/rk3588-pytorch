/* Training-mode BatchNorm on the CPU (NCHW fp32), on the library's thread pool. Included from rknpu_gemm.c.
 *
 * PyTorch's aarch64 BatchNorm reductions run through generic kernels: on MobileNet / EfficientNet-style
 * models the forward and backward of the BatchNorm layers were 20-25% of a training step once the convolutions
 * ran on the NPU. Here each channel is one work item: the channel's N x HW slab (N runs of HW contiguous floats)
 * is read once for the statistics and then again, while still in the L2 cache, to normalize.
 *
 *   op 0: forward  y = (x - mean) * rstd * gamma + beta; writes mean, rstd (1/sqrt(var + eps)) and var (biased)
 *   op 1: backward dx from g, x, gamma, mean, rstd; dgamma, dbeta (any of gamma / dgamma / dbeta may be NULL)
 *
 * Sums are accumulated per run in four fp32 NEON lanes and added up in double, so the statistics match
 * PyTorch's to ~1e-7 relative.
 */
#include <arm_neon.h>

typedef struct {
	int op;
	const float *x, *g, *gamma, *beta, *mean_in, *rstd_in;
	float *y, *mean, *rstd, *var, *dgamma, *dbeta;
	int N, C, HW;
	float eps;
} bn_job;

/* sum of p[0..n) and, with c0 given, of (p - c0)^2 or (p-c0)*q */
static inline double bn_sum(const float *p, int n) {
	int i = 0;
	float32x4_t a = vdupq_n_f32(0.f), b = vdupq_n_f32(0.f);
	for (; i + 8 <= n; i += 8) { a = vaddq_f32(a, vld1q_f32(p + i)); b = vaddq_f32(b, vld1q_f32(p + i + 4)); }
	double s = (double)vaddvq_f32(vaddq_f32(a, b));
	for (; i < n; i++) s += p[i];
	return s;
}
static inline double bn_sumsq_c(const float *p, int n, float c0) {
	int i = 0;
	float32x4_t a = vdupq_n_f32(0.f), b = vdupq_n_f32(0.f), c = vdupq_n_f32(c0);
	for (; i + 8 <= n; i += 8) {
		float32x4_t u = vsubq_f32(vld1q_f32(p + i), c), v = vsubq_f32(vld1q_f32(p + i + 4), c);
		a = vfmaq_f32(a, u, u);
		b = vfmaq_f32(b, v, v);
	}
	double s = (double)vaddvq_f32(vaddq_f32(a, b));
	for (; i < n; i++) { float u = p[i] - c0; s += (double)u * u; }
	return s;
}
/* sum g and sum g * (x - c0) */
static inline void bn_sum_g(const float *g, const float *x, int n, float c0, double *sg, double *sgx) {
	int i = 0;
	float32x4_t a = vdupq_n_f32(0.f), b = vdupq_n_f32(0.f), c = vdupq_n_f32(c0);
	for (; i + 4 <= n; i += 4) {
		float32x4_t gv = vld1q_f32(g + i);
		a = vaddq_f32(a, gv);
		b = vfmaq_f32(b, gv, vsubq_f32(vld1q_f32(x + i), c));
	}
	double s1 = (double)vaddvq_f32(a), s2 = (double)vaddvq_f32(b);
	for (; i < n; i++) { s1 += g[i]; s2 += (double)g[i] * (x[i] - c0); }
	*sg += s1;
	*sgx += s2;
}
/* y = k1 * a + k2 * b + k3 (b may be NULL: k2 ignored) */
static inline void bn_axpby(float *y, const float *a, const float *b, int n, float k1, float k2, float k3) {
	int i = 0;
	float32x4_t v3 = vdupq_n_f32(k3);
	if (b) {
		for (; i + 4 <= n; i += 4) {
			float32x4_t r = vfmaq_n_f32(v3, vld1q_f32(a + i), k1);
			vst1q_f32(y + i, vfmaq_n_f32(r, vld1q_f32(b + i), k2));
		}
		for (; i < n; i++) y[i] = k1 * a[i] + k2 * b[i] + k3;
	} else {
		for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vfmaq_n_f32(v3, vld1q_f32(a + i), k1));
		for (; i < n; i++) y[i] = k1 * a[i] + k3;
	}
}

/* A channel's N runs are C*HW floats apart: with power-of-two strides they all land in the same few L2 sets and
 * evict each other, so a second pass over the slab would come from DRAM again. The first pass therefore also copies
 * the slab into a contiguous per-thread buffer, and the later passes read that (L2 resident). */
#define BN_STAGE_MAX (192 * 1024)       /* bytes per staged array */
static __thread float *bn_tl_buf[2];
static __thread size_t bn_tl_cap;
static int bn_stage_get(size_t floats, float **a, float **b) {
	if (floats * 4 > BN_STAGE_MAX) return 0;
	if (bn_tl_cap < floats) {
		free(bn_tl_buf[0]); free(bn_tl_buf[1]);
		bn_tl_buf[0] = malloc(floats * 4);
		bn_tl_buf[1] = b ? malloc(floats * 4) : NULL;
		bn_tl_cap = (bn_tl_buf[0] && (!b || bn_tl_buf[1])) ? floats : 0;
		if (!bn_tl_cap) return 0;
	} else if (b && !bn_tl_buf[1]) {
		bn_tl_buf[1] = malloc(bn_tl_cap * 4);
		if (!bn_tl_buf[1]) return 0;
	}
	*a = bn_tl_buf[0];
	if (b) *b = bn_tl_buf[1];
	return 1;
}
/* sum of p[0..n) while copying it to dst */
static inline double bn_sum_copy(const float *p, float *dst, int n) {
	int i = 0;
	float32x4_t a = vdupq_n_f32(0.f), b = vdupq_n_f32(0.f);
	for (; i + 8 <= n; i += 8) {
		float32x4_t u = vld1q_f32(p + i), v = vld1q_f32(p + i + 4);
		vst1q_f32(dst + i, u); vst1q_f32(dst + i + 4, v);
		a = vaddq_f32(a, u); b = vaddq_f32(b, v);
	}
	double s = (double)vaddvq_f32(vaddq_f32(a, b));
	for (; i < n; i++) { dst[i] = p[i]; s += p[i]; }
	return s;
}
/* sum g and sum g * (x - c0) while copying both */
static inline void bn_sum_g_copy(const float *g, const float *x, float *gd, float *xd, int n, float c0, double *sg, double *sgx) {
	int i = 0;
	float32x4_t a = vdupq_n_f32(0.f), b = vdupq_n_f32(0.f), c = vdupq_n_f32(c0);
	for (; i + 4 <= n; i += 4) {
		float32x4_t gv = vld1q_f32(g + i), xv = vld1q_f32(x + i);
		vst1q_f32(gd + i, gv); vst1q_f32(xd + i, xv);
		a = vaddq_f32(a, gv);
		b = vfmaq_f32(b, gv, vsubq_f32(xv, c));
	}
	double s1 = (double)vaddvq_f32(a), s2 = (double)vaddvq_f32(b);
	for (; i < n; i++) { gd[i] = g[i]; xd[i] = x[i]; s1 += g[i]; s2 += (double)g[i] * (x[i] - c0); }
	*sg += s1;
	*sgx += s2;
}

static void bn_range(void *v, int b, int e) {
	const bn_job *J = v;
	const int N = J->N, C = J->C, HW = J->HW;
	const double M = (double)N * HW;
	for (int c = b; c < e; c++) {
		const long stride = (long)C * HW;
		const float *xc = J->x + (long)c * HW;
		if (J->op == 0) {
			float *sx = NULL;
			const int staged = bn_stage_get((size_t)N * HW, &sx, NULL);
			double s = 0;
			for (int n = 0; n < N; n++)
				s += staged ? bn_sum_copy(xc + n * stride, sx + (long)n * HW, HW) : bn_sum(xc + n * stride, HW);
			const float mean = (float)(s / M);
			double ss = 0;
			for (int n = 0; n < N; n++)
				ss += bn_sumsq_c(staged ? sx + (long)n * HW : xc + n * stride, HW, mean);
			const float var = (float)(ss / M), rstd = 1.f / sqrtf(var + J->eps);
			J->mean[c] = mean;
			J->rstd[c] = rstd;
			J->var[c] = var;
			const float ga = J->gamma ? J->gamma[c] : 1.f, be = J->beta ? J->beta[c] : 0.f;
			const float k1 = ga * rstd, k3 = be - mean * k1;
			for (int n = 0; n < N; n++)
				bn_axpby(J->y + (long)c * HW + n * stride, staged ? sx + (long)n * HW : xc + n * stride, NULL, HW, k1, 0.f, k3);
		} else {
			const float mean = J->mean_in[c], rstd = J->rstd_in[c];
			const float *gc = J->g + (long)c * HW;
			double sg = 0, sgx = 0;
			float *sxb = NULL, *sgb = NULL;
			const int staged = J->y && bn_stage_get((size_t)N * HW, &sxb, &sgb);
			for (int n = 0; n < N; n++) {
				if (staged) bn_sum_g_copy(gc + n * stride, xc + n * stride, sgb + (long)n * HW, sxb + (long)n * HW, HW, mean, &sg, &sgx);
				else bn_sum_g(gc + n * stride, xc + n * stride, HW, mean, &sg, &sgx);
			}
			if (J->dbeta) J->dbeta[c] = (float)sg;
			if (J->dgamma) J->dgamma[c] = (float)(sgx * rstd);
			if (!J->y) continue;
			const float gr = (J->gamma ? J->gamma[c] : 1.f) * rstd;
			/* dx = gr * (g - sg/M - (x - mean) * rstd^2 * sgx/M) = k1 g + k2 x + k3 */
			const float k1 = gr, k2 = (float)(-gr * rstd * rstd * (sgx / M)), k3 = (float)(-gr * (sg / M)) - k2 * mean;
			for (int n = 0; n < N; n++)
				bn_axpby(J->y + (long)c * HW + n * stride, staged ? sgb + (long)n * HW : gc + n * stride,
				         staged ? sxb + (long)n * HW : xc + n * stride, HW, k1, k2, k3);
		}
	}
}

int rk_batchnorm(rk_ctx *ctx, int op, const float *x, const float *g, const float *gamma, const float *beta,
                 const float *mean_in, const float *rstd_in, float *y, float *mean, float *rstd, float *var,
                 float *dgamma, float *dbeta, int N, int C, int HW, float eps) {
	if (op != 0 && op != 1) return -1;
	if (N <= 0 || C <= 0 || HW <= 0) return -1;
	bn_job J = { .op = op, .x = x, .g = g, .gamma = gamma, .beta = beta, .mean_in = mean_in, .rstd_in = rstd_in,
	             .y = y, .mean = mean, .rstd = rstd, .var = var, .dgamma = dgamma, .dbeta = dbeta,
	             .N = N, .C = C, .HW = HW, .eps = eps };
	parallel_for(ctx, C, 1, bn_range, &J);
	return 0;
}
