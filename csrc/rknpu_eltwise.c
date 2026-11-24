/*
 * Fused element-wise kernels for the CPU side of a training step (included by rknpu_gemm.c, which provides
 * the thread pool). PyTorch's aarch64 wheels run these ops through generic kernels ("CPU capability: DEFAULT"):
 * GELU (erf) forward + backward on 2048 x 1536 fp32 takes 56 ms on three A76 cores, RMSNorm 29 ms (it is
 * composed of four ops). One pass each, NEON, fp32 math, fp32 or bf16 tensors.
 *
 * Accuracy: exp() to ~2 ulp (Cephes polynomial), erf() to 1.5e-7 absolute (Abramowitz & Stegun 7.1.26),
 * so GELU / SiLU agree with PyTorch's to ~3e-7 relative; RMSNorm is exact up to summation order.
 */
#include <math.h>

enum {
	EW_GELU_FWD = 0, EW_GELU_BWD = 1, EW_GELU_TANH_FWD = 2, EW_GELU_TANH_BWD = 3,
	EW_SILU_FWD = 4, EW_SILU_BWD = 5, EW_RMS_FWD = 6, EW_RMS_BWD = 7,
};

typedef struct {
	int op, bf16;
	const void *x, *g, *w;
	void *y;
	float *aux;          /* RMS: per-row rstd (fwd output, bwd input) */
	float *dw_part;      /* RMS bwd: [blocks][D] partial weight gradients */
	long n, rows, D;
	float eps;
} ew_job;

#define EW_BLOCK 4096    /* elements per work item for the pointwise ops */
#define EW_ROWS 32       /* rows per work item for RMSNorm */

#if HAVE_NEON8
static inline float32x4_t exp_ps(float32x4_t x) {
	const float32x4_t hi = vdupq_n_f32(88.3762626647949f), lo = vdupq_n_f32(-87.3365447505531f);
	x = vminq_f32(vmaxq_f32(x, lo), hi);
	float32x4_t fx = vrndnq_f32(vmulq_n_f32(x, 1.44269504088896341f));
	x = vfmsq_f32(x, fx, vdupq_n_f32(0.693359375f));
	x = vfmsq_f32(x, fx, vdupq_n_f32(-2.12194440e-4f));
	float32x4_t z = vmulq_f32(x, x);
	float32x4_t y = vdupq_n_f32(1.9875691500e-4f);
	y = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), y, x);
	y = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), y, x);
	y = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), y, x);
	y = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), y, x);
	y = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), y, x);
	y = vfmaq_f32(vaddq_f32(x, vdupq_n_f32(1.f)), y, z);
	int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(127)), 23);
	return vmulq_f32(y, vreinterpretq_f32_s32(e));
}
/* erf(z) for z >= 0 given e = exp(-z^2) (A&S 7.1.26) */
static inline float32x4_t erf_pos(float32x4_t az, float32x4_t e) {
	float32x4_t t = vdivq_f32(vdupq_n_f32(1.f), vfmaq_n_f32(vdupq_n_f32(1.f), az, 0.3275911f));
	float32x4_t p = vdupq_n_f32(1.061405429f);
	p = vfmaq_f32(vdupq_n_f32(-1.453152027f), p, t);
	p = vfmaq_f32(vdupq_n_f32(1.421413741f), p, t);
	p = vfmaq_f32(vdupq_n_f32(-0.284496736f), p, t);
	p = vfmaq_f32(vdupq_n_f32(0.254829592f), p, t);
	return vfmsq_f32(vdupq_n_f32(1.f), vmulq_f32(p, t), e);
}
static inline float32x4_t ld4(const void *p, long i, int bf) {
	if (bf) return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16((const uint16_t *)p + i), 16));
	return vld1q_f32((const float *)p + i);
}
static inline void st4(void *p, long i, float32x4_t v, int bf) {
	if (bf) vst1_u16((uint16_t *)p + i, bf16x4(v));
	else vst1q_f32((float *)p + i, v);
}
/* the op on 4 lanes: x, g -> y */
static inline float32x4_t ew4(int op, float32x4_t x, float32x4_t g) {
	const float32x4_t one = vdupq_n_f32(1.f), half = vdupq_n_f32(0.5f);
	switch (op) {
	case EW_GELU_FWD: case EW_GELU_BWD: {
		float32x4_t z = vmulq_n_f32(x, 0.70710678118654752f), az = vabsq_f32(z);
		float32x4_t e = exp_ps(vnegq_f32(vmulq_f32(az, az)));             /* exp(-x^2/2) */
		float32x4_t er = erf_pos(az, e);
		er = vbslq_f32(vcltq_f32(z, vdupq_n_f32(0.f)), vnegq_f32(er), er);
		float32x4_t cdf = vmulq_f32(half, vaddq_f32(one, er));
		if (op == EW_GELU_FWD) return vmulq_f32(x, cdf);
		float32x4_t pdf = vmulq_n_f32(e, 0.39894228040143268f);
		return vmulq_f32(g, vfmaq_f32(cdf, x, pdf));
	}
	case EW_GELU_TANH_FWD: case EW_GELU_TANH_BWD: {
		float32x4_t x2 = vmulq_f32(x, x);
		float32x4_t u = vmulq_n_f32(vfmaq_n_f32(x, vmulq_f32(x2, x), 0.044715f), 0.79788456080286536f);
		/* tanh(u) = 1 - 2 / (exp(2u) + 1), saturating correctly for large |u| */
		float32x4_t th = vsubq_f32(one, vdivq_f32(vdupq_n_f32(2.f), vaddq_f32(exp_ps(vaddq_f32(u, u)), one)));
		if (op == EW_GELU_TANH_FWD) return vmulq_f32(vmulq_f32(half, x), vaddq_f32(one, th));
		float32x4_t du = vmulq_n_f32(vfmaq_n_f32(one, x2, 3.f * 0.044715f), 0.79788456080286536f);
		float32x4_t d = vfmaq_f32(vmulq_f32(half, vaddq_f32(one, th)),
		                          vmulq_f32(vmulq_f32(half, x), vfmsq_f32(one, th, th)), du);
		return vmulq_f32(g, d);
	}
	default: {   /* SiLU */
		float32x4_t s = vdivq_f32(one, vaddq_f32(one, exp_ps(vnegq_f32(x))));
		if (op == EW_SILU_FWD) return vmulq_f32(x, s);
		return vmulq_f32(g, vmulq_f32(s, vfmaq_f32(one, x, vsubq_f32(one, s))));
	}
	}
}
#endif

static float ew1(int op, float x, float g) {
	switch (op) {
	case EW_GELU_FWD: return 0.5f * x * (1.f + erff(x * 0.70710678118654752f));
	case EW_GELU_BWD: return g * (0.5f * (1.f + erff(x * 0.70710678118654752f)) + x * expf(-0.5f * x * x) * 0.39894228040143268f);
	case EW_GELU_TANH_FWD: case EW_GELU_TANH_BWD: {
		float u = 0.79788456080286536f * (x + 0.044715f * x * x * x), th = tanhf(u);
		if (op == EW_GELU_TANH_FWD) return 0.5f * x * (1.f + th);
		return g * (0.5f * (1.f + th) + 0.5f * x * (1.f - th * th) * 0.79788456080286536f * (1.f + 3.f * 0.044715f * x * x));
	}
	case EW_SILU_FWD: return x / (1.f + expf(-x));
	default: { float s = 1.f / (1.f + expf(-x)); return g * s * (1.f + x * (1.f - s)); }
	}
}

static inline float ldf(const void *p, long i, int bf) {
	if (!bf) return ((const float *)p)[i];
	uint32_t u = (uint32_t)((const uint16_t *)p)[i] << 16;
	float f;
	memcpy(&f, &u, 4);
	return f;
}
static inline void stf(void *p, long i, float v, int bf) {
	if (bf) ((uint16_t *)p)[i] = f32_to_bf16(v);
	else ((float *)p)[i] = v;
}

static void ew_point_range(void *v, int b, int e) {
	ew_job *J = v;
	for (int it = b; it < e; it++) {
		long i0 = (long)it * EW_BLOCK, i1 = i0 + EW_BLOCK < J->n ? i0 + EW_BLOCK : J->n, i = i0;
#if HAVE_NEON8
		const float32x4_t z = vdupq_n_f32(0.f);
		for (; i + 4 <= i1; i += 4)
			st4(J->y, i, ew4(J->op, ld4(J->x, i, J->bf16), J->g ? ld4(J->g, i, J->bf16) : z), J->bf16);
#endif
		for (; i < i1; i++) stf(J->y, i, ew1(J->op, ldf(J->x, i, J->bf16), J->g ? ldf(J->g, i, J->bf16) : 0.f), J->bf16);
	}
}

/* RMSNorm over the last dim: y = x * rstd * w, rstd = 1 / sqrt(mean(x^2) + eps) */
static void ew_rms_range(void *v, int b, int e) {
	ew_job *J = v;
	const long D = J->D;
	const int bf = J->bf16;
	for (int it = b; it < e; it++) {
		long r0 = (long)it * EW_ROWS, r1 = r0 + EW_ROWS < J->rows ? r0 + EW_ROWS : J->rows;
		float *dwp = J->dw_part ? J->dw_part + (size_t)it * D : NULL;
		if (dwp) memset(dwp, 0, (size_t)D * 4);
		for (long r = r0; r < r1; r++) {
			long base = r * D, d = 0;
			if (J->op == EW_RMS_FWD) {
				double ss = 0;
#if HAVE_NEON8
				float32x4_t acc = vdupq_n_f32(0.f);
				for (; d + 4 <= D; d += 4) { float32x4_t a = ld4(J->x, base + d, bf); acc = vfmaq_f32(acc, a, a); }
				ss = vaddvq_f32(acc);
#endif
				for (; d < D; d++) { float a = ldf(J->x, base + d, bf); ss += a * a; }
				float rs = 1.f / sqrtf((float)(ss / D) + J->eps);
				J->aux[r] = rs;
				d = 0;
#if HAVE_NEON8
				for (; d + 4 <= D; d += 4) {
					float32x4_t a = vmulq_n_f32(ld4(J->x, base + d, bf), rs);
					if (J->w) a = vmulq_f32(a, ld4(J->w, d, bf));
					st4(J->y, base + d, a, bf);
				}
#endif
				for (; d < D; d++) stf(J->y, base + d, ldf(J->x, base + d, bf) * rs * (J->w ? ldf(J->w, d, bf) : 1.f), bf);
			} else {
				/* dx = rs * (gw - x * rs^2 * mean(gw * x)), gw = g * w; dw += g * x * rs */
				const float rs = J->aux[r];
				double dot = 0;
#if HAVE_NEON8
				float32x4_t acc = vdupq_n_f32(0.f);
				for (; d + 4 <= D; d += 4) {
					float32x4_t gw = ld4(J->g, base + d, bf);
					if (J->w) gw = vmulq_f32(gw, ld4(J->w, d, bf));
					acc = vfmaq_f32(acc, gw, ld4(J->x, base + d, bf));
				}
				dot = vaddvq_f32(acc);
#endif
				for (; d < D; d++) dot += ldf(J->g, base + d, bf) * (J->w ? ldf(J->w, d, bf) : 1.f) * ldf(J->x, base + d, bf);
				const float c = rs * rs * rs * (float)(dot / D);
				d = 0;
#if HAVE_NEON8
				for (; d + 4 <= D; d += 4) {
					float32x4_t gg = ld4(J->g, base + d, bf), xx = ld4(J->x, base + d, bf);
					float32x4_t gw = J->w ? vmulq_f32(gg, ld4(J->w, d, bf)) : gg;
					st4(J->y, base + d, vfmsq_n_f32(vmulq_n_f32(gw, rs), xx, c), bf);
					if (dwp) vst1q_f32(dwp + d, vfmaq_f32(vld1q_f32(dwp + d), gg, vmulq_n_f32(xx, rs)));
				}
#endif
				for (; d < D; d++) {
					float gg = ldf(J->g, base + d, bf), xx = ldf(J->x, base + d, bf);
					float gw = J->w ? gg * ldf(J->w, d, bf) : gg;
					stf(J->y, base + d, gw * rs - xx * c, bf);
					if (dwp) dwp[d] += gg * xx * rs;
				}
			}
		}
	}
}

typedef struct { float *part, *out; int blocks; long D; } dw_job;
static void dw_sum_range(void *v, int b, int e) {
	dw_job *W = v;
	for (int it = b; it < e; it++) {
		long d0 = (long)it * 256, d1 = d0 + 256 < W->D ? d0 + 256 : W->D;
		for (long d = d0; d < d1; d++) {
			float s = 0.f;
			for (int k = 0; k < W->blocks; k++) s += W->part[(size_t)k * W->D + d];
			W->out[d] = s;
		}
	}
}

/*
 * op: EW_* ; bf16: tensors x, g, y, w are bf16 (else fp32); aux (rstd) and dw are always fp32.
 * Pointwise ops: n = rows * D elements. RMS: rows x D, w may be NULL, dw (bwd, optional) gets sum over rows.
 */
int rk_eltwise(rk_ctx *ctx, int op, int bf16, const void *x, const void *g, const void *w, void *y, float *aux,
               float *dw, long rows, long D, float eps) {
	ew_job J = { .op = op, .bf16 = bf16, .x = x, .g = g, .w = w, .y = y, .aux = aux, .n = rows * D, .rows = rows,
	             .D = D, .eps = eps };
	if (op <= EW_SILU_BWD) {
		parallel_for(ctx, (int)((J.n + EW_BLOCK - 1) / EW_BLOCK), 4, ew_point_range, &J);
		return 0;
	}
	if (op != EW_RMS_FWD && op != EW_RMS_BWD) return -1;
	int blocks = (int)((rows + EW_ROWS - 1) / EW_ROWS);
	float *part = NULL;
	if (op == EW_RMS_BWD && dw) {
		part = malloc((size_t)blocks * D * 4);
		if (!part) return -1;
		J.dw_part = part;
	}
	parallel_for(ctx, blocks, 1, ew_rms_range, &J);
	if (part) {
		dw_job W = { part, dw, blocks, D };
		parallel_for(ctx, (int)((D + 255) / 256), 1, dw_sum_range, &W);
		free(part);
	}
	return 0;
}
