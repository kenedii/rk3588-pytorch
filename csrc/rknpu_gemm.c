/*
 * rknpu_gemm: fp16 GEMM on the RK3588 NPU through the vendor rknpu DRM driver.
 *
 * A matmul is expressed as a 1x1 convolution: A (M x K) becomes a feature map of
 * height M, width 1 and K channels; B (K x N) becomes N kernels of K channels; the
 * output is a feature map of height M and N channels. The convolution buffer (CBUF,
 * 12 banks x 32 KB per core) must hold the feature-map tile, so A is split into row
 * tiles and K is split into chunks (partial sums are reduced on the CPU in fp32).
 *
 * Register sequence and data layouts follow the reverse engineering in
 * https://github.com/mtx512/rk3588-npu (GPL-3.0).
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <drm/drm.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "npu_regs.h"
#include "rknpu_gemm.h"
#include "rknpu_uapi.h"

#define REGCMD_SLOTS   112   /* uint64 words reserved per task (108 used) */
#define REGCFG_AMOUNT  104   /* registers before the 4-word PC tail */
#define PC_TASK_LIMIT  4000  /* PC task counter is 12 bits on RK3588 */

typedef _Float16 f16;

typedef struct {
	uint32_t handle;
	uint64_t obj;
	uint64_t dma;
	void    *ptr;
	size_t   size;
	uint32_t flags;
} rk_buf;

struct rk_ctx {
	int      fd;
	int      mode;
	int      threads;
	int      max_m;
	int      max_k;
	int      max_n;
	int      cached;
	int      timeout_ms;
	rk_buf   regcmd, tasks, a, b, c;
	rk_stats stats;
	char     err[256];
};

static double now_us(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static void set_err(rk_ctx *ctx, const char *fmt, ...) {
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(ctx->err, sizeof(ctx->err), fmt, ap);
	va_end(ap);
}

const char *rk_last_error(rk_ctx *ctx) { return ctx ? ctx->err : "no context"; }

static int is_rknpu(int fd) {
	char name[64] = {0};
	struct drm_version v;
	memset(&v, 0, sizeof(v));
	v.name = name;
	v.name_len = sizeof(name) - 1;
	if (ioctl(fd, DRM_IOCTL_VERSION, &v) < 0) return 0;
	return strcmp(name, "rknpu") == 0;
}

static int env_int(const char *name, int dflt) {
	const char *s = getenv(name);
	return (s && *s) ? atoi(s) : dflt;
}

rk_ctx *rk_open(const char *dev) {
	int fd = -1;
	if (!dev) dev = getenv("RKNPU_DEV");
	if (dev) {
		fd = open(dev, O_RDWR | O_CLOEXEC);
		if (fd >= 0 && !is_rknpu(fd)) { close(fd); fd = -1; }
	} else {
		for (int i = 0; i < 16 && fd < 0; i++) {
			char path[32];
			snprintf(path, sizeof(path), "/dev/dri/card%d", i);
			fd = open(path, O_RDWR | O_CLOEXEC);
			if (fd >= 0 && !is_rknpu(fd)) { close(fd); fd = -1; }
		}
	}
	if (fd < 0) return NULL;
	rk_ctx *ctx = calloc(1, sizeof(*ctx));
	ctx->fd = fd;
	ctx->mode = env_int("RKNPU_MODE", 1);
	ctx->threads = env_int("RKNPU_THREADS", 3);
	ctx->max_m = env_int("RKNPU_MAX_M", 256);
	ctx->max_k = env_int("RKNPU_MAX_K", 4096);
	ctx->max_n = env_int("RKNPU_MAX_N", 4096);
	ctx->cached = env_int("RKNPU_CACHED", 1);
	ctx->timeout_ms = env_int("RKNPU_TIMEOUT_MS", 5000);
	return ctx;
}

static void buf_free(rk_ctx *ctx, rk_buf *b) {
	if (b->ptr) munmap(b->ptr, b->size);
	if (b->handle) {
		struct rknpu_mem_destroy d = { .handle = b->handle, .obj_addr = b->obj };
		ioctl(ctx->fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &d);
	}
	memset(b, 0, sizeof(*b));
}

void rk_close(rk_ctx *ctx) {
	if (!ctx) return;
	buf_free(ctx, &ctx->regcmd);
	buf_free(ctx, &ctx->tasks);
	buf_free(ctx, &ctx->a);
	buf_free(ctx, &ctx->b);
	buf_free(ctx, &ctx->c);
	close(ctx->fd);
	free(ctx);
}

void rk_set_mode(rk_ctx *ctx, int mode) { ctx->mode = mode; }
void rk_set_threads(rk_ctx *ctx, int t) { ctx->threads = t > 0 ? t : 1; }
void rk_set_max_m_tile(rk_ctx *ctx, int m) { ctx->max_m = m >= 4 ? m & ~3 : 4; }
void rk_get_stats(rk_ctx *ctx, rk_stats *out) { *out = ctx->stats; }

/* Grow-only buffer: reallocates when the request exceeds the current size. */
static int buf_ensure(rk_ctx *ctx, rk_buf *b, size_t size, uint32_t flags) {
	if (b->ptr && b->size >= size && b->flags == flags) return 0;
	buf_free(ctx, b);
	size = (size + 4095) & ~(size_t)4095;
	if (size < 65536) size = 65536;
	/* this driver keeps per-core IOMMU mappings: map every buffer on all three cores */
	struct rknpu_mem_create mc = { .flags = flags, .size = size, .core_mask = 0x7 };
	if (ioctl(ctx->fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mc) < 0) {
		set_err(ctx, "MEM_CREATE(%zu, 0x%x) failed: %s", size, flags, strerror(errno));
		return -1;
	}
	if (mc.dma_addr + size > 0xffffffffull) {
		set_err(ctx, "NPU buffer above 4 GB IOVA (0x%llx)", (unsigned long long)mc.dma_addr);
		struct rknpu_mem_destroy d = { .handle = mc.handle, .obj_addr = mc.obj_addr };
		ioctl(ctx->fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &d);
		return -1;
	}
	struct rknpu_mem_map mm = { .handle = mc.handle };
	if (ioctl(ctx->fd, DRM_IOCTL_RKNPU_MEM_MAP, &mm) < 0) {
		set_err(ctx, "MEM_MAP failed: %s", strerror(errno));
		return -1;
	}
	void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, ctx->fd, mm.offset);
	if (p == MAP_FAILED) {
		set_err(ctx, "mmap of NPU buffer failed: %s", strerror(errno));
		struct rknpu_mem_destroy d = { .handle = mc.handle, .obj_addr = mc.obj_addr };
		ioctl(ctx->fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &d);
		return -1;
	}
	b->handle = mc.handle;
	b->obj = mc.obj_addr;
	b->dma = mc.dma_addr;
	b->ptr = p;
	b->size = size;
	b->flags = flags;
	return 0;
}

static void buf_sync(rk_ctx *ctx, rk_buf *b, size_t size, uint32_t dir) {
	if (!(b->flags & RKNPU_MEM_CACHEABLE)) return;
	struct rknpu_mem_sync s = { .flags = dir, .obj_addr = b->obj, .offset = 0, .size = size };
	ioctl(ctx->fd, DRM_IOCTL_RKNPU_MEM_SYNC, &s);
}

/* One CNA+CORE+DPU task computing out[Mt x N] = in[Mt x K] * w[N x K]^T (fp16 in, fp32 out). */
static void emit_task(uint64_t *o, int m, int k, int n, uint32_t in_dma, uint32_t w_dma,
                      uint32_t out_dma, uint32_t next_dma, int dbanks) {
	int i = 0;
	/* CNA DMA strides as observed in vendor command streams: line = width*4, surface = line*(H/4 - 1) */
	uint32_t surf_stride = 4u * (uint32_t)(m / 4 - 1);
	uint32_t wbytes_kernel = (uint32_t)k * 2u;
	uint32_t entries = (uint32_t)((k + 31) / 32);

	o[i++] = npuop(OP_REG_DPU, 0xE, DPU_S_POINTER);
	o[i++] = npuop(OP_REG_CNA, (PREC_FP16 << 7) | (PREC_FP16 << 4), CNA_CONV_CON1);
	o[i++] = npuop(OP_REG_CNA, ((uint32_t)(m + 1) & 0x3FF) << 4, CNA_CONV_CON2);
	o[i++] = npuop(OP_REG_CNA, (1 << 3) | 1, CNA_CONV_CON3);
	o[i++] = npuop(OP_REG_CNA, (1u << 16) | ((uint32_t)m & 0x7FF), CNA_DATA_SIZE0);
	o[i++] = npuop(OP_REG_CNA, (((uint32_t)k - 1) << 16) | (uint32_t)k, CNA_DATA_SIZE1);
	o[i++] = npuop(OP_REG_CNA, 1, CNA_DATA_SIZE2);
	o[i++] = npuop(OP_REG_CNA, (uint32_t)m & 0x3FFFF, CNA_DATA_SIZE3);
	o[i++] = npuop(OP_REG_CNA, wbytes_kernel * (uint32_t)n, CNA_WEIGHT_SIZE0);
	o[i++] = npuop(OP_REG_CNA, wbytes_kernel & 0x7FFFF, CNA_WEIGHT_SIZE1);
	o[i++] = npuop(OP_REG_CNA, (1u << 24) | (1u << 16) | ((uint32_t)n & 0x3FFF), CNA_WEIGHT_SIZE2);
	o[i++] = npuop(OP_REG_CNA, ((CBUF_BANKS - dbanks) << 4) | (uint32_t)dbanks, CNA_CBUF_CON0);
	o[i++] = npuop(OP_REG_CNA, entries & 0x1FFF, CNA_CBUF_CON1);
	o[i++] = npuop(OP_REG_CNA, (1 << 3) | (1 << 1) | 1, CNA_CVT_CON0);
	o[i++] = npuop(OP_REG_CNA, 1u << 16, CNA_CVT_CON1);
	o[i++] = npuop(OP_REG_CNA, 1u << 16, CNA_CVT_CON2);
	o[i++] = npuop(OP_REG_CNA, 1u << 16, CNA_CVT_CON3);
	o[i++] = npuop(OP_REG_CNA, 1u << 16, CNA_CVT_CON4);
	o[i++] = npuop(OP_REG_CNA, 0, CNA_FC_CON0);
	o[i++] = npuop(OP_REG_CNA, 0, CNA_FC_CON1);
	o[i++] = npuop(OP_REG_CNA, 0, CNA_PAD_CON0);
	o[i++] = npuop(OP_REG_CNA, in_dma, CNA_FEATURE_DATA_ADDR);
	o[i++] = npuop(OP_REG_CNA, 0, CNA_FC_CON2);
	o[i++] = npuop(OP_REG_CNA, (15u << 16) | 15u, CNA_DMA_CON0);
	o[i++] = npuop(OP_REG_CNA, 4, CNA_DMA_CON1);
	o[i++] = npuop(OP_REG_CNA, surf_stride & 0xFFFFFFF, CNA_DMA_CON2);
	o[i++] = npuop(OP_REG_CNA, (1u << 16) | ((uint32_t)m & 0x7FF), CNA_FC_DATA_SIZE0);
	o[i++] = npuop(OP_REG_CNA, (uint32_t)k & 0xFFFF, CNA_FC_DATA_SIZE1);
	o[i++] = npuop(OP_REG_CNA, 0, CNA_DCOMP_CTRL);
	o[i++] = npuop(OP_REG_CNA, 0, CNA_DCOMP_REGNUM);
	o[i++] = npuop(OP_REG_CNA, w_dma, CNA_DCOMP_ADDR0);
	for (int r = 0; r < 16; r++) o[i++] = npuop(OP_REG_CNA, 0, CNA_DCOMP_AMOUNT0 + 4 * r);
	o[i++] = npuop(OP_REG_CNA, 0, CNA_CVT_CON5);
	o[i++] = npuop(OP_REG_CNA, 0, CNA_PAD_CON1);

	o[i++] = npuop(OP_REG_CORE, (PREC_FP16 << 8) | 1, CORE_MISC_CFG);
	o[i++] = npuop(OP_REG_CORE, ((uint32_t)(m - 1) << 16), CORE_DATAOUT_SIZE_0);
	o[i++] = npuop(OP_REG_CORE, (uint32_t)(n - 1), CORE_DATAOUT_SIZE_1);
	o[i++] = npuop(OP_REG_CORE, 0, CORE_CLIP_TRUNCATE);
	o[i++] = npuop(OP_REG_CORE, 0, CORE_3030);

	o[i++] = npuop(OP_REG_DPU, (15u << 5) | (2u << 1), DPU_FEATURE_MODE_CFG);
	o[i++] = npuop(OP_REG_DPU, ((uint32_t)PREC_FP32 << 29) | (PREC_FP16 << 26) | PREC_FP16, DPU_DATA_FORMAT);
	o[i++] = npuop(OP_REG_DPU, 0, DPU_OFFSET_PEND);
	o[i++] = npuop(OP_REG_DPU, out_dma, DPU_DST_BASE_ADD);
	o[i++] = npuop(OP_REG_DPU, ((uint32_t)m & 0xFFFFFFF) << 4, DPU_DST_SURF_STRIDE);
	o[i++] = npuop(OP_REG_DPU, 0, DPU_DATA_CUBE_WIDTH);
	o[i++] = npuop(OP_REG_DPU, (uint32_t)(m - 1) & 0x1FFF, DPU_DATA_CUBE_HEIGHT);
	o[i++] = npuop(OP_REG_DPU, 0, DPU_DATA_CUBE_NOTCH_ADDR);
	o[i++] = npuop(OP_REG_DPU, (((uint32_t)(n - 1) & 0x1FFF) << 16) | ((uint32_t)(n - 1) & 0x1FFF), DPU_DATA_CUBE_CHANNEL);
	o[i++] = npuop(OP_REG_DPU, (1 << 6) | (1 << 4) | (1 << 1) | 1, DPU_BS_CFG);
	o[i++] = npuop(OP_REG_DPU, 0, DPU_BS_ALU_CFG);
	o[i++] = npuop(OP_REG_DPU, 0, DPU_BS_MUL_CFG);
	o[i++] = npuop(OP_REG_DPU, 0, DPU_BS_RELUX_CMP_VALUE);
	o[i++] = npuop(OP_REG_DPU, (3 << 8) | (3 << 5) | (3 << 2) | (1 << 1), DPU_BS_OW_CFG);
	o[i++] = npuop(OP_REG_DPU, 0, DPU_BS_OW_OP);
	o[i++] = npuop(OP_REG_DPU, (uint32_t)(n - 1) & 0x1FFF, DPU_WDMA_SIZE_0);
	o[i++] = npuop(OP_REG_DPU, ((uint32_t)(m - 1) & 0x1FFF) << 16, DPU_WDMA_SIZE_1);
	o[i++] = npuop(OP_REG_DPU, (1 << 6) | (1 << 4) | (1 << 1) | 1, DPU_BN_CFG);
	o[i++] = npuop(OP_REG_DPU, 0, DPU_BN_ALU_CFG);
	o[i++] = npuop(OP_REG_DPU, 0, DPU_BN_MUL_CFG);
	o[i++] = npuop(OP_REG_DPU, 0, DPU_BN_RELUX_CMP_VALUE);
	o[i++] = npuop(OP_REG_DPU, (1 << 9) | (1 << 8) | (1 << 7) | (1 << 1) | 1, DPU_EW_CFG);
	o[i++] = npuop(OP_REG_DPU, 0, DPU_EW_CVT_OFFSET_VALUE);
	o[i++] = npuop(OP_REG_DPU, 1, DPU_EW_CVT_SCALE_VALUE);
	o[i++] = npuop(OP_REG_DPU, 0, DPU_EW_RELUX_CMP_VALUE);
	o[i++] = npuop(OP_REG_DPU, 0, DPU_OUT_CVT_OFFSET);
	o[i++] = npuop(OP_REG_DPU, 1, DPU_OUT_CVT_SCALE);
	o[i++] = npuop(OP_REG_DPU, 0, DPU_OUT_CVT_SHIFT);
	for (int r = 0; r < 8; r++) o[i++] = npuop(OP_REG_DPU, 0, DPU_EW_OP_VALUE_0 + 4 * r);
	o[i++] = npuop(OP_REG_DPU, (((uint32_t)m * 4u) & 0xFFFFFFF) << 4, DPU_SURFACE_ADD);
	o[i++] = npuop(OP_REG_DPU, 0, DPU_40C4);
	for (int r = 0; r < 12; r++) o[i++] = npuop(OP_REG_DPU, 0, DPU_LUT_ACCESS_CFG + 4 * r);
	/* i == 104: PC tail. Chain to the next task's register commands if there is one. */
	if (next_dma) {
		o[i++] = npuop(OP_REG_PC, next_dma, PC_BASE_ADDRESS);
		o[i++] = npuop(OP_REG_PC, (REGCFG_AMOUNT + 4 + 1) / 2 - 1, PC_REGISTER_AMOUNTS);
	} else {
		o[i++] = 0;
		o[i++] = npuop(OP_REG_PC, 0, PC_REGISTER_AMOUNTS);
	}
	o[i++] = npuop(OP_40, 0, 0);
	o[i++] = npuop(OP_ENABLE, PC_ENABLE_DPU | PC_ENABLE_CNA | PC_ENABLE, PC_OPERATION_ENABLE);
}

static int submit(rk_ctx *ctx, int start, int count, int cores) {
	struct rknpu_submit s;
	memset(&s, 0, sizeof(s));
	s.flags = RKNPU_JOB_PC | RKNPU_JOB_PINGPONG;
	s.timeout = ctx->timeout_ms;
	s.task_start = start;
	s.task_number = count;
	s.task_obj_addr = ctx->tasks.obj;
	s.task_base_addr = ctx->tasks.dma;
	s.fence_fd = -1;
	if (cores <= 1) {
		s.core_mask = 1;
		s.subcore_task[0].task_start = start;
		s.subcore_task[0].task_number = count;
	} else {
		/* split [start, start+count) into `cores` contiguous ranges, one per core */
		s.core_mask = (1u << cores) - 1;
		int base = start;
		for (int c = 0; c < cores; c++) {
			int n = count / cores + (c < count % cores);
			s.subcore_task[c].task_start = base;
			s.subcore_task[c].task_number = n;
			base += n;
		}
	}
	__sync_synchronize();
	double t0 = now_us();
	int r = ioctl(ctx->fd, DRM_IOCTL_RKNPU_SUBMIT, &s);
	ctx->stats.submit_us += now_us() - t0;
	ctx->stats.submits++;
	if (r < 0) {
		set_err(ctx, "SUBMIT(start=%d, n=%d, cores=%d) failed: %s", start, count, cores, strerror(errno));
		return -1;
	}
	ctx->stats.hw_us += s.hw_elapse_time / 1000.0; /* driver reports ns */
	return 0;
}

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
static inline int round_up(int a, int b) { return ceil_div(a, b) * b; }

int rk_gemm_f32(rk_ctx *ctx, int M, int N, int K,
                const float *A, int64_t sa_m, int64_t sa_k,
                const float *B, int64_t sb_k, int64_t sb_n,
                float *C, int64_t ldc, int accumulate) {
	double t_start = now_us();
	memset(&ctx->stats, 0, sizeof(ctx->stats));
	if (M <= 0 || N <= 0 || K <= 0) { set_err(ctx, "bad shape"); return -1; }

	/* ---- tiling plan ---- */
	int Kp = round_up(K, 32);
	int nk = ceil_div(Kp, ctx->max_k);
	int Kc = round_up(ceil_div(Kp, nk), 32);
	int Np = round_up(N, 16);
	int nn = ceil_div(Np, ctx->max_n);
	int Nc = round_up(ceil_div(Np, nn), 16);
	/* keep at least one 16-kernel weight group resident in CBUF */
	int wbanks = ceil_div(16 * Kc * 2, CBUF_BANK_SIZE);
	if (wbanks < 1) wbanks = 1;
	int dbanks_max = CBUF_BANKS - wbanks;
	if (dbanks_max < 1) { set_err(ctx, "K chunk %d too large for CBUF", Kc); return -1; }
	int Mt = (int)((dbanks_max * (long)CBUF_BANK_SIZE) / (Kc * 2)) & ~3;
	if (Mt > ctx->max_m) Mt = ctx->max_m & ~3;
	if (Mt > round_up(M, 4)) Mt = round_up(M, 4);
	if (Mt < 4) { set_err(ctx, "K chunk %d leaves no room for 4 rows", Kc); return -1; }
	int nm = ceil_div(M, Mt);
	Mt = round_up(ceil_div(M, nm), 4);              /* rebalance tiles */
	int dbanks = ceil_div(Mt * Kc * 2, CBUF_BANK_SIZE);
	int T = nm * nk * nn;
	ctx->stats.m_tile = Mt; ctx->stats.k_tile = Kc; ctx->stats.n_tile = Nc;
	ctx->stats.tasks = T;

	size_t a_tile = (size_t)Mt * Kc;   /* fp16 elements */
	size_t b_tile = (size_t)Nc * Kc;   /* fp16 elements */
	size_t c_tile = (size_t)Mt * Nc;   /* fp32 elements */
	size_t a_bytes = a_tile * nm * nk * 2, b_bytes = b_tile * nk * nn * 2, c_bytes = c_tile * T * 4;
	uint32_t dflags = (ctx->cached ? RKNPU_MEM_CACHEABLE : RKNPU_MEM_WRITE_COMBINE) | RKNPU_MEM_IOMMU | RKNPU_MEM_NON_CONTIGUOUS;
	if (buf_ensure(ctx, &ctx->a, a_bytes, dflags) ||
	    buf_ensure(ctx, &ctx->b, b_bytes, dflags) ||
	    buf_ensure(ctx, &ctx->c, c_bytes, dflags) ||
	    buf_ensure(ctx, &ctx->regcmd, (size_t)T * REGCMD_SLOTS * 8, RKNPU_MEM_WRITE_COMBINE | RKNPU_MEM_IOMMU) ||
	    buf_ensure(ctx, &ctx->tasks, (size_t)T * sizeof(struct rknpu_task), RKNPU_MEM_WRITE_COMBINE | RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_IOMMU))
		return -1;

	f16 *ap = ctx->a.ptr, *bp = ctx->b.ptr;
	float *cp = ctx->c.ptr;
	int nthr = ctx->threads;

	/* ---- pack A: tile (i,j) laid out as [Kc/8][Mt][8] (NC1HWC2, C2=8) ---- */
	double t0 = now_us();
	#pragma omp parallel for collapse(2) schedule(static) num_threads(nthr)
	for (int ij = 0; ij < nm * nk; ij++) {
		for (int kb = 0; kb < Kc / 8; kb++) {
			int i = ij / nk, j = ij % nk;
			int m0 = i * Mt, k0 = j * Kc + kb * 8;
			f16 *dst = ap + (size_t)ij * a_tile + (size_t)kb * Mt * 8;
			for (int m = 0; m < Mt; m++) {
				int gm = m0 + m;
				f16 *d = dst + m * 8;
				if (gm < M && k0 + 8 <= K) {
					const float *s = A + gm * sa_m + (int64_t)k0 * sa_k;
					if (sa_k == 1) { for (int t = 0; t < 8; t++) d[t] = (f16)s[t]; }
					else { for (int t = 0; t < 8; t++) d[t] = (f16)s[t * sa_k]; }
				} else {
					for (int t = 0; t < 8; t++) {
						int gk = k0 + t;
						d[t] = (gm < M && gk < K) ? (f16)A[gm * sa_m + (int64_t)gk * sa_k] : (f16)0;
					}
				}
			}
		}
	}
	/* ---- pack B: tile (j,c) laid out as [Nc/16][Kc/32][16][32] ---- */
	#pragma omp parallel for collapse(2) schedule(static) num_threads(nthr)
	for (int jc = 0; jc < nk * nn; jc++) {
		for (int ng = 0; ng < Nc / 16; ng++) {
			int j = jc / nn, c = jc % nn;
			f16 *dst = bp + (size_t)jc * b_tile + (size_t)ng * 16 * Kc;
			for (int cb = 0; cb < Kc / 32; cb++) {
				for (int nr = 0; nr < 16; nr++) {
					int gn = c * Nc + ng * 16 + nr;
					f16 *d = dst + cb * 512 + nr * 32;
					int k0 = j * Kc + cb * 32;
					if (gn < N && k0 + 32 <= K) {
						const float *s = B + (int64_t)k0 * sb_k + (int64_t)gn * sb_n;
						if (sb_k == 1) { for (int t = 0; t < 32; t++) d[t] = (f16)s[t]; }
						else { for (int t = 0; t < 32; t++) d[t] = (f16)s[t * sb_k]; }
					} else {
						for (int t = 0; t < 32; t++) {
							int gk = k0 + t;
							d[t] = (gn < N && gk < K) ? (f16)B[(int64_t)gk * sb_k + (int64_t)gn * sb_n] : (f16)0;
						}
					}
				}
			}
		}
	}
	buf_sync(ctx, &ctx->a, a_bytes, RKNPU_MEM_SYNC_TO_DEVICE);
	buf_sync(ctx, &ctx->b, b_bytes, RKNPU_MEM_SYNC_TO_DEVICE);
	ctx->stats.pack_us = now_us() - t0;

	/* ---- build register command streams and task descriptors ---- */
	uint64_t *rc = ctx->regcmd.ptr;
	struct rknpu_task *tk = ctx->tasks.ptr;
	int cores = ctx->mode == 2 ? 3 : 1;
	/* chain boundaries: a task links to the next unless it ends a submit/core range */
	int per_submit = ctx->mode == 0 ? 1 : PC_TASK_LIMIT * cores;
	for (int t = 0; t < T; t++) {
		int c = t % nn, ij = t / nn, j = ij % nk, i = ij / nk;
		uint32_t in_dma = (uint32_t)(ctx->a.dma + ((size_t)(i * nk + j) * a_tile) * 2);
		uint32_t w_dma = (uint32_t)(ctx->b.dma + ((size_t)(j * nn + c) * b_tile) * 2);
		uint32_t out_dma = (uint32_t)(ctx->c.dma + (size_t)t * c_tile * 4);
		int ncols = (c * Nc + Nc <= Np) ? Nc : Np - c * Nc;
		/* is t the last task of its chain? */
		int sub_start = (t / per_submit) * per_submit;
		int sub_count = (T - sub_start < per_submit) ? T - sub_start : per_submit;
		int last = 0;
		if (ctx->mode == 0) last = 1;
		else {
			int base = sub_start;
			for (int cc = 0; cc < cores; cc++) {
				int n = sub_count / cores + (cc < sub_count % cores);
				if (n > 0 && t == base + n - 1) last = 1;
				base += n;
			}
		}
		uint32_t next = last ? 0 : (uint32_t)(ctx->regcmd.dma + (size_t)(t + 1) * REGCMD_SLOTS * 8);
		emit_task(rc + (size_t)t * REGCMD_SLOTS, Mt, Kc, ncols, in_dma, w_dma, out_dma, next, dbanks);
		struct rknpu_task d = {
			.flags = 0, .op_idx = 0, .enable_mask = 0xd, .int_mask = 0x300, .int_clear = 0x1ffff,
			.int_status = 0, .regcfg_amount = REGCFG_AMOUNT, .regcfg_offset = 0,
			.regcmd_addr = ctx->regcmd.dma + (size_t)t * REGCMD_SLOTS * 8,
		};
		tk[t] = d;
	}

	/* ---- run ---- */
	for (int s = 0; s < T; s += per_submit) {
		int cnt = (T - s < per_submit) ? T - s : per_submit;
		int cc = cores;
		if (cnt < cc) cc = 1;   /* not enough tasks to split */
		if (cc == 1 && cores > 1) {
			/* re-link: a single-core range must be one chain */
			for (int t = s; t < s + cnt - 1; t++) {
				uint64_t *o = rc + (size_t)t * REGCMD_SLOTS;
				o[104] = npuop(OP_REG_PC, (uint32_t)(ctx->regcmd.dma + (size_t)(t + 1) * REGCMD_SLOTS * 8), PC_BASE_ADDRESS);
				o[105] = npuop(OP_REG_PC, (REGCFG_AMOUNT + 4 + 1) / 2 - 1, PC_REGISTER_AMOUNTS);
			}
		}
		if (submit(ctx, s, cnt, cc) < 0) return -1;
	}

	/* ---- unpack: out tile layout [Nc/4][Mt][4], reduce over K chunks ---- */
	t0 = now_us();
	buf_sync(ctx, &ctx->c, c_bytes, RKNPU_MEM_SYNC_FROM_DEVICE);
	#pragma omp parallel for collapse(2) schedule(static) num_threads(nthr)
	for (int i = 0; i < nm; i++) {
		for (int m = 0; m < Mt; m++) {
			int gm = i * Mt + m;
			if (gm >= M) continue;
			float *crow = C + (int64_t)gm * ldc;
			for (int c = 0; c < nn; c++) {
				int n0 = c * Nc;
				int ncols = N - n0 < Nc ? N - n0 : Nc;
				for (int n = 0; n < ncols; n++) {
					size_t off = (size_t)(n >> 2) * Mt * 4 + (size_t)m * 4 + (n & 3);
					float acc = 0.f;
					for (int j = 0; j < nk; j++) {
						int t = (i * nk + j) * nn + c;
						acc += cp[(size_t)t * c_tile + off];
					}
					if (accumulate) crow[n0 + n] += acc; else crow[n0 + n] = acc;
				}
			}
		}
	}
	ctx->stats.unpack_us = now_us() - t0;
	ctx->stats.total_us = now_us() - t_start;
	return 0;
}
