/*
 * rknpu_gemm: fp16 matrix multiply on the RK3588 NPU without the RKNN runtime.
 *
 * C[M,N] = A[M,K] @ B[K,N], fp32 in/out, fp16 multiply, fp32 accumulate/output.
 * A and B are read through arbitrary element strides, so transposed operands
 * (needed for backward passes) cost nothing extra.
 */
#ifndef RKNPU_GEMM_H
#define RKNPU_GEMM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rk_ctx rk_ctx;

typedef struct {
	double pack_us;     /* fp32 -> fp16 conversion + NPU layout packing (A and B) */
	double submit_us;   /* wall time inside the submit ioctl(s) */
	double hw_us;       /* NPU execution time reported by the driver */
	double unpack_us;   /* NPU layout -> row-major fp32 (+ K-split reduction) */
	double total_us;
	int    tasks;       /* NPU tasks in the last call */
	int    submits;     /* submit ioctls in the last call */
	int    m_tile, k_tile, n_tile;
	double sync_us;     /* CPU cache maintenance (MEM_SYNC ioctls) */
	int    cache_hits;  /* weight operands taken from the packed-weight cache */
	int    waves;       /* pipeline waves of the last GEMM (1 = not pipelined; pack/unpack then overlap submit) */
} rk_stats;

/* Opens the NPU DRM node (NULL = auto-detect the "rknpu" card). Returns NULL on failure. */
rk_ctx *rk_open(const char *dev);
void    rk_close(rk_ctx *ctx);
const char *rk_last_error(rk_ctx *ctx);

/* Execution mode (default 2): 0 = one submit per task (single core),
 *                 1 = all tasks chained in one submit (single core),
 *                 2 = chained, split across the three NPU cores. */
void rk_set_mode(rk_ctx *ctx, int mode);
void rk_set_threads(rk_ctx *ctx, int threads);   /* CPU threads for packing (default 3) */
void rk_set_max_m_tile(rk_ctx *ctx, int m);      /* rows of A per NPU task (default 1020) */
void rk_set_max_buf_mb(rk_ctx *ctx, int mb);     /* cap NPU buffers per call; bigger GEMMs are split */

/*
 * C = A @ B (or C += A @ B if accumulate).
 * Element A(m,k) is A[m*sa_m + k*sa_k], B(k,n) is B[k*sb_k + n*sb_n], C(m,n) is C[m*ldc + n].
 * Returns 0 on success, negative on failure (see rk_last_error).
 */
int rk_gemm_f32(rk_ctx *ctx, int M, int N, int K,
                const float *A, int64_t sa_m, int64_t sa_k,
                const float *B, int64_t sb_k, int64_t sb_n,
                float *C, int64_t ldc, int accumulate);

void rk_get_stats(rk_ctx *ctx, rk_stats *out);

/* Precisions for rk_gemm_ex (all take and return fp32):
 *   RK_PREC_FP16        fp16 x fp16, fp32 accumulate/output, power-of-two auto-scaling (default)
 *   RK_PREC_FP16_OUT16  fp16 x fp16 -> fp16 output (no auto-scaling; outputs must fit fp16)
 *   RK_PREC_INT8        int8 x int8 -> int32, per-row (A) / per-column (B) symmetric quantization
 *   RK_PREC_FP16X3      split fp32 into fp16 hi + lo parts; 3 products, close to fp32 accuracy
 *   RK_PREC_BF16        bf16 x bf16 -> fp32 (only if the hardware probe confirms it) */
enum { RK_PREC_FP16 = 0, RK_PREC_FP16_OUT16 = 1, RK_PREC_INT8 = 2, RK_PREC_FP16X3 = 3, RK_PREC_BF16 = 4 };

int rk_gemm_ex(rk_ctx *ctx, int prec, int M, int N, int K,
               const float *A, int64_t sa_m, int64_t sa_k,
               const float *B, int64_t sb_k, int64_t sb_n,
               float *C, int64_t ldc, int accumulate);

/* Batched: C[b] = A[b] @ B[b] for b < batch; batch strides in elements (0 = shared operand).
 * All batch elements go to the NPU as one chain of tasks. */
int rk_gemm_batched(rk_ctx *ctx, int prec, int batch, int M, int N, int K,
                    const float *A, int64_t sa_b, int64_t sa_m, int64_t sa_k,
                    const float *B, int64_t sb_b, int64_t sb_k, int64_t sb_n,
                    float *C, int64_t sc_b, int64_t ldc, int accumulate);

/* Free all NPU buffers (they are re-created on demand) / bytes currently held. */
void   rk_release(rk_ctx *ctx);
size_t rk_buffer_bytes(rk_ctx *ctx);

typedef struct {
	int driver_version;     /* e.g. 908 = 0.9.8 */
	uint32_t hw_version;
	int iommu;
	uint32_t freq_hz;
} rk_info;
int rk_query(rk_ctx *ctx, rk_info *info);

/* Fused CPU element-wise ops (NEON, the library's thread pool), fp32 or bf16 (bf16 = 1) tensors:
 * op 0/1 GELU fwd/bwd, 2/3 GELU-tanh fwd/bwd, 4/5 SiLU fwd/bwd (y = op(x[, g]) over rows*D elements),
 * 6/7 RMSNorm fwd/bwd over rows x D (w optional; aux = per-row rstd, fp32; dw = fp32 weight grad, optional). */
int rk_eltwise(rk_ctx *ctx, int op, int bf16, const void *x, const void *g, const void *w, void *y, float *aux,
               float *dw, long rows, long D, float eps);

/* Hardware probing: one task, raw bytes, caller-chosen precision codes (M%4==0, K%32==0, N%16==0;
 * a_rows is M x K row-major, w_bytes is N*K elements in the NPU weight layout). */
typedef struct {
	int in_prec, proc_prec, out_prec, in_bytes, out_bytes, kgroup, qd_en;
	uint32_t bs_ow, cvt_scale, cvt_shift, surf_add;
} rk_probe_cfg;
int rk_probe_raw(rk_ctx *ctx, const rk_probe_cfg *cfg, int M, int N, int K,
                 const void *a_rows, const void *w_bytes, void *c_rows);

/* 2-D convolution on the NPU (fp32 NCHW in/out; prec RK_PREC_FP16 or RK_PREC_BF16).
 * x: (N, C, H, W), w: (O, C, kh, kw), y: (N, O, oh, ow); padding top/left = pt/pl, bottom/right
 * implied by oh/ow. Dilation is not supported (see rk_conv2d_grouped for groups). */
int rk_conv2d(rk_ctx *ctx, int prec, int N, int C, int H, int W, const float *x,
              int O, int kh, int kw, const float *w, int sy, int sx, int pt, int pl,
              int oh, int ow, float *y, int accumulate);

/* Grouped convolution: w is (O, C/groups, kh, kw); groups must divide C and O. */
int rk_conv2d_grouped(rk_ctx *ctx, int prec, int N, int C, int H, int W, const float *x,
                      int O, int kh, int kw, const float *w, int sy, int sx, int pt, int pl,
                      int oh, int ow, float *y, int accumulate, int groups);

/* Output epilogue for the following GEMM / convolution calls (sticky until changed):
 * y = relu?(y + bias), bias indexed by output column (GEMM) or output channel (conv);
 * bias = NULL for none. The ReLU runs in the NPU's DPU when nothing is added afterwards. */
void rk_set_epilogue(rk_ctx *ctx, const float *bias, int relu);

/* Packed-weight cache. While a nonzero key is set, the weight operand (GEMM B, conv
 * weights) of the following calls is packed once into a dedicated NPU buffer under that key
 * and reused by later calls with the same key, precision and shape. The caller guarantees
 * that a key always names the same weight values (change the key when they change).
 * LRU-evicted beyond RKNPU_WCACHE_MB (default 256; 0 disables). */
void   rk_set_wcache_key(rk_ctx *ctx, uint64_t key);
void   rk_wcache_drop(rk_ctx *ctx, uint64_t key);
void   rk_wcache_clear(rk_ctx *ctx);
size_t rk_wcache_bytes(rk_ctx *ctx);
int    rk_wcache_entries(rk_ctx *ctx);
void   rk_set_wcache_mb(rk_ctx *ctx, int mb);

/* Raw single-task convolution (fp16 NHWC in, fp32 NHWC out) for probing. */
typedef struct {
	int h, w, c, n, kh, kw, sy, sx, pt, pl, oh, ow;
	int grains, dbanks;     /* 0 = defaults */
	int layout;             /* 0 = row-major (NHWC) attempt, 1 = NC1HWC2 */
	int dw;                 /* depthwise (w_bytes = one kernel per channel) */
	int relu;               /* DPU ReLU */
} rk_conv_cfg;
int rk_conv_raw(rk_ctx *ctx, const rk_conv_cfg *cfg, const void *in_nhwc, const void *w_bytes, void *out_nhwc);

#ifdef __cplusplus
}
#endif
#endif
