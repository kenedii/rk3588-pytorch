/*
 * rknpu_gemm: fp16 matrix multiply on the RK3588 NPU without the RKNN runtime.
 *
 * C[M,N] = A[M,K] @ B[K,N], fp32 in/out, fp16 multiply, fp32 accumulate/output.
 * A and B are read through arbitrary element strides, so transposed operands
 * (needed for backward passes) cost nothing extra.
 */
#ifndef RKNPU_GEMM_H
#define RKNPU_GEMM_H

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
} rk_stats;

/* Opens the NPU DRM node (NULL = auto-detect the "rknpu" card). Returns NULL on failure. */
rk_ctx *rk_open(const char *dev);
void    rk_close(rk_ctx *ctx);
const char *rk_last_error(rk_ctx *ctx);

/* Execution mode: 0 = one submit per task (single core),
 *                 1 = all tasks chained in one submit (single core),
 *                 2 = chained, split across the three NPU cores. */
void rk_set_mode(rk_ctx *ctx, int mode);
void rk_set_threads(rk_ctx *ctx, int threads);   /* CPU threads for packing (default 3) */
void rk_set_max_m_tile(rk_ctx *ctx, int m);      /* rows of A per NPU task (default 256) */

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

#ifdef __cplusplus
}
#endif
#endif
