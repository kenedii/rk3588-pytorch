/* Depthwise 2-D convolution on the CPU (NCHW fp32, one filter per channel), on the library's thread pool.
 *
 * The NPU's depthwise mode doesn't work in fp16 yet (raxda re docs/07), and PyTorch's CPU depthwise backward is
 * very slow on the A76/A55 (e.g. 295 ms for 32x256x32x32 3x3, against ~50 ms forward), so MobileNet-style models
 * spent most of a training step there. These loops are written as row-segment multiply-adds that GCC vectorizes;
 * the weight gradient uses explicit NEON dot products. Included from rknpu_gemm.c.
 *
 *   op 0: y  = dwconv(x, w)                  (y: N,C,OH,OW)
 *   op 1: dx = dwconv_transpose(dy, w)       (dx: N,C,H,W)
 *   op 2: dw = sum_n corr(dy, x)             (dw: C,KH,KW)
 */
#include <arm_neon.h>

typedef struct {
	const float *x, *w, *dy;
	float *y;
	int N, C, H, W, KH, KW, sy, sx, pt, pl, OH, OW;
} dwc_job;

/* output columns ox with 0 <= ox*s - p + j < W */
static inline void dwc_cols(int j, int s, int p, int W, int OW, int *lo, int *hi) {
	int a = p - j;
	int l = a > 0 ? (a + s - 1) / s : 0;
	int h = (W - 1 + p - j) >= 0 ? (W - 1 + p - j) / s + 1 : 0;
	*lo = l;
	*hi = h < OW ? h : OW;
}

static void dwc_fwd_range(void *v, int b, int e) {
	const dwc_job *J = v;
	const long ip = (long)J->H * J->W, op = (long)J->OH * J->OW;
	for (int pc = b; pc < e; pc++) {
		const int c = pc % J->C;
		const float *x = J->x + pc * ip, *w = J->w + (long)c * J->KH * J->KW;
		float *y = J->y + pc * op;
		memset(y, 0, op * sizeof(float));
		for (int j = 0; j < J->KW; j++) {
			int lo, hi;
			dwc_cols(j, J->sx, J->pl, J->W, J->OW, &lo, &hi);
			if (lo >= hi) continue;
			for (int oy = 0; oy < J->OH; oy++) {
				float *yr = y + (long)oy * J->OW;
				for (int i = 0; i < J->KH; i++) {
					const int iy = oy * J->sy - J->pt + i;
					if (iy < 0 || iy >= J->H) continue;
					const float wv = w[i * J->KW + j];
					const float *xr = x + (long)iy * J->W - J->pl + j;
					if (J->sx == 1)
						for (int ox = lo; ox < hi; ox++) yr[ox] += wv * xr[ox];
					else
						for (int ox = lo; ox < hi; ox++) yr[ox] += wv * xr[ox * J->sx];
				}
			}
		}
	}
}

static void dwc_bwd_in_range(void *v, int b, int e) {
	const dwc_job *J = v;
	const long ip = (long)J->H * J->W, op = (long)J->OH * J->OW;
	for (int pc = b; pc < e; pc++) {
		const int c = pc % J->C;
		const float *dy = J->dy + pc * op, *w = J->w + (long)c * J->KH * J->KW;
		float *dx = J->y + pc * ip;
		memset(dx, 0, ip * sizeof(float));
		for (int j = 0; j < J->KW; j++) {
			int lo, hi;
			dwc_cols(j, J->sx, J->pl, J->W, J->OW, &lo, &hi);
			if (lo >= hi) continue;
			for (int oy = 0; oy < J->OH; oy++) {
				const float *gr = dy + (long)oy * J->OW;
				for (int i = 0; i < J->KH; i++) {
					const int iy = oy * J->sy - J->pt + i;
					if (iy < 0 || iy >= J->H) continue;
					const float wv = w[i * J->KW + j];
					float *xr = dx + (long)iy * J->W - J->pl + j;
					if (J->sx == 1)
						for (int ox = lo; ox < hi; ox++) xr[ox] += wv * gr[ox];
					else
						for (int ox = lo; ox < hi; ox++) xr[ox * J->sx] += wv * gr[ox];
				}
			}
		}
	}
}

static inline float dwc_dot(const float *a, const float *b, int n) {
	float32x4_t s0 = vdupq_n_f32(0.f), s1 = vdupq_n_f32(0.f);
	int k = 0;
	for (; k + 8 <= n; k += 8) {
		s0 = vfmaq_f32(s0, vld1q_f32(a + k), vld1q_f32(b + k));
		s1 = vfmaq_f32(s1, vld1q_f32(a + k + 4), vld1q_f32(b + k + 4));
	}
	float s = vaddvq_f32(vaddq_f32(s0, s1));
	for (; k < n; k++) s += a[k] * b[k];
	return s;
}

/* one channel per index: dw[c] accumulated in double across rows/batch for accuracy on big maps */
static void dwc_bwd_w_range(void *v, int b, int e) {
	const dwc_job *J = v;
	const long ip = (long)J->H * J->W, op = (long)J->OH * J->OW;
	const int KK = J->KH * J->KW;
	for (int c = b; c < e; c++) {
		double acc[KK];
		for (int k = 0; k < KK; k++) acc[k] = 0.0;
		for (int n = 0; n < J->N; n++) {
			const long pc = (long)n * J->C + c;
			const float *x = J->x + pc * ip, *dy = J->dy + pc * op;
			for (int j = 0; j < J->KW; j++) {
				int lo, hi;
				dwc_cols(j, J->sx, J->pl, J->W, J->OW, &lo, &hi);
				if (lo >= hi) continue;
				for (int i = 0; i < J->KH; i++) {
					float s = 0.f;
					for (int oy = 0; oy < J->OH; oy++) {
						const int iy = oy * J->sy - J->pt + i;
						if (iy < 0 || iy >= J->H) continue;
						const float *gr = dy + (long)oy * J->OW;
						const float *xr = x + (long)iy * J->W - J->pl + j;
						if (J->sx == 1)
							s += dwc_dot(gr + lo, xr + lo, hi - lo);
						else
							for (int ox = lo; ox < hi; ox++) s += gr[ox] * xr[ox * J->sx];
					}
					acc[i * J->KW + j] += s;
				}
			}
		}
		float *dw = J->y + (long)c * KK;
		for (int k = 0; k < KK; k++) dw[k] = (float)acc[k];
	}
}

int rk_dwconv(rk_ctx *ctx, int op, const float *x, const float *w, const float *dy, float *out,
              int N, int C, int H, int W, int KH, int KW, int sy, int sx, int pt, int pl, int OH, int OW) {
	if (!ctx || N <= 0 || C <= 0 || KH <= 0 || KW <= 0 || sy <= 0 || sx <= 0 || OH <= 0 || OW <= 0) return -1;
	dwc_job J = { .x = x, .w = w, .dy = dy, .y = out, .N = N, .C = C, .H = H, .W = W, .KH = KH, .KW = KW,
	             .sy = sy, .sx = sx, .pt = pt, .pl = pl, .OH = OH, .OW = OW };
	switch (op) {
	case 0: parallel_for(ctx, N * C, 1, dwc_fwd_range, &J); return 0;
	case 1: parallel_for(ctx, N * C, 1, dwc_bwd_in_range, &J); return 0;
	case 2: parallel_for(ctx, C, 1, dwc_bwd_w_range, &J); return 0;
	}
	return -1;
}
