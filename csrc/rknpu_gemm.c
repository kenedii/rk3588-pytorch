/*
 * rknpu_gemm: fp16 GEMM on the RK3588 NPU through the vendor rknpu DRM driver,
 * without the RKNN runtime.
 *
 * A matmul runs as a 1x1 convolution: A (M x K) is a feature map of height M and
 * K channels, B (K x N) is N kernels of K channels, C is M x N. The CNA is put in
 * row-major feature mode (CONV_CON1 bit 29, line stride K/8, surface stride 0), so
 * it reads A as a plain row-major fp16 matrix and the DPU writes C as a plain
 * row-major fp32 matrix: the only layout conversion left on the CPU is packing B
 * into kernel groups.
 *
 * The convolution buffer (CBUF, 12 banks x 32 KB per core) holds a tile of A rows
 * plus at least one 16-kernel weight group; A is split into row tiles, one task per
 * tile, chained through the PC. Very large K is split into chunks whose partial sums
 * are reduced on the CPU in fp32.
 *
 * The register stream layout was recovered by capturing librknnrt's rknn_matmul
 * command streams with tools/ioctl_spy.c; the original register map and kernel
 * layout come from https://github.com/mtx512/rk3588-npu (GPL-3.0).
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
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <sys/file.h>

#include "npu_regs.h"
#include "rknpu_gemm.h"
#include "rknpu_uapi.h"

#define REGCMD_SLOTS   112   /* uint64 words per task: 108 registers + 4-word PC tail */
#define REGCFG_AMOUNT  108
#define PC_AMOUNT      ((REGCFG_AMOUNT + 4 + 1) / 2 - 1)  /* 128-bit fetch units - 1 */
#define PC_TASK_LIMIT  4000  /* PC task counter is 12 bits on RK3588 */
#define MEM_FLAG_400   (1u << 10)  /* set by librknnrt on every allocation */
#define CBUF_WEIGHT_REUSE (1u << 13)

typedef _Float16 f16;
struct pool;
static void pool_destroy(struct pool *p);

typedef struct {
	uint32_t handle;
	uint64_t obj;
	uint64_t dma;
	void    *ptr;
	size_t   size;
	uint32_t flags;
} rk_buf;

/* One cached, already packed weight operand (GEMM B or convolution weights) */
struct wc_entry {
	uint64_t key;
	int      kind;              /* 1 = GEMM B, 2 = convolution weights */
	int      prec;
	int      d[6];              /* shape: GEMM (K, N); conv (O, Cg, kh, kw, groups) */
	int      layout[4];         /* GEMM tiling (Kc, Nc, nk, nn) the bytes were packed for */
	int      packed;
	rk_buf   buf;
	float    scale;             /* fp16 auto-scale of the operand */
	float   *col;               /* int8: per-column scales [N] then their inverses [N] */
	uint64_t last_use;
};

struct rk_ctx {
	int      fd;
	int      mode;
	int      threads;
	int      max_m;
	int      max_k;
	int      max_n;
	int      timeout_ms;
	int      reuse;
	struct pool *pool;
	int      autoscale;
	int      lock_fd;          /* cross-process lock (flock) or -1 */
	size_t   max_buf;          /* cap on NPU buffer bytes per call; larger GEMMs are split */
	float   *scr[3];
	size_t   scr_n[3];
	rk_buf   regcmd, tasks, a, b, c;
	/* output epilogue for the next calls (see rk_set_epilogue) */
	const float *epi_bias;
	int      epi_relu;
	/* packed-weight cache (see rk_set_wcache_key) */
	uint64_t wc_key;
	struct wc_entry *wc;
	int      wc_n, wc_cap;
	size_t   wc_bytes, wc_max;
	uint64_t wc_clock;
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
	ctx->mode = env_int("RKNPU_MODE", 2);
	ctx->threads = env_int("RKNPU_THREADS", 3);
	ctx->max_m = env_int("RKNPU_MAX_M", 1020);
	ctx->max_k = env_int("RKNPU_MAX_K", 8192);
	ctx->max_n = env_int("RKNPU_MAX_N", 4096);
	ctx->timeout_ms = env_int("RKNPU_TIMEOUT_MS", 3000);
	ctx->reuse = env_int("RKNPU_WEIGHT_REUSE", 1);
	ctx->autoscale = env_int("RKNPU_AUTOSCALE", 1);
	ctx->max_buf = (size_t)env_int("RKNPU_MAX_BUF_MB", 512) << 20;
	ctx->wc_max = (size_t)env_int("RKNPU_WCACHE_MB", 256) << 20;
	ctx->lock_fd = -1;
	if (env_int("RKNPU_LOCK", 1)) {
		const char *lp = getenv("RKNPU_LOCK_FILE") ? getenv("RKNPU_LOCK_FILE") : "/tmp/rk3588_torch.lock";
		ctx->lock_fd = open(lp, O_RDWR | O_CREAT | O_CLOEXEC, 0666);
	}

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

static void wc_free_entry(rk_ctx *ctx, struct wc_entry *e) {
	buf_free(ctx, &e->buf);
	free(e->col);
	ctx->wc_bytes -= e->buf.size;
	memset(e, 0, sizeof(*e));
}

void rk_wcache_clear(rk_ctx *ctx) {
	for (int i = 0; i < ctx->wc_n; i++) wc_free_entry(ctx, &ctx->wc[i]);
	ctx->wc_n = 0;
	ctx->wc_bytes = 0;
}

void rk_wcache_drop(rk_ctx *ctx, uint64_t key) {
	for (int i = 0; i < ctx->wc_n; i++)
		if (ctx->wc[i].key == key) {
			wc_free_entry(ctx, &ctx->wc[i]);
			ctx->wc[i] = ctx->wc[--ctx->wc_n];
			i--;
		}
}

size_t rk_wcache_bytes(rk_ctx *ctx) { return ctx->wc_bytes; }
int    rk_wcache_entries(rk_ctx *ctx) { return ctx->wc_n; }
void   rk_set_wcache_mb(rk_ctx *ctx, int mb) { ctx->wc_max = (size_t)(mb > 0 ? mb : 0) << 20; }
void   rk_set_wcache_key(rk_ctx *ctx, uint64_t key) { ctx->wc_key = key; }
void   rk_set_epilogue(rk_ctx *ctx, const float *bias, int relu) { ctx->epi_bias = bias; ctx->epi_relu = relu; }

/* Find the entry for (key, kind, prec, shape); NULL if absent. */
static struct wc_entry *wc_find(rk_ctx *ctx, int kind, int prec, const int *d, int nd) {
	if (!ctx->wc_key || ctx->wc_max == 0) return NULL;
	for (int i = 0; i < ctx->wc_n; i++) {
		struct wc_entry *e = &ctx->wc[i];
		if (e->key == ctx->wc_key && e->kind == kind && e->prec == prec && !memcmp(e->d, d, nd * sizeof(int))) {
			e->last_use = ++ctx->wc_clock;
			return e;
		}
	}
	return NULL;
}

/* New (unpacked) entry; evicts least recently used ones to stay under the byte cap. */
static struct wc_entry *wc_new(rk_ctx *ctx, int kind, int prec, const int *d, int nd, size_t bytes) {
	if (!ctx->wc_key || ctx->wc_max == 0 || bytes > ctx->wc_max) return NULL;
	rk_wcache_drop(ctx, ctx->wc_key);            /* stale shapes for this key */
	while (ctx->wc_n > 0 && ctx->wc_bytes + bytes > ctx->wc_max) {
		int lru = 0;
		for (int i = 1; i < ctx->wc_n; i++) if (ctx->wc[i].last_use < ctx->wc[lru].last_use) lru = i;
		wc_free_entry(ctx, &ctx->wc[lru]);
		ctx->wc[lru] = ctx->wc[--ctx->wc_n];
	}
	if (ctx->wc_n == ctx->wc_cap) {
		int cap = ctx->wc_cap ? ctx->wc_cap * 2 : 16;
		struct wc_entry *n = realloc(ctx->wc, cap * sizeof(*n));
		if (!n) return NULL;
		ctx->wc = n;
		ctx->wc_cap = cap;
	}
	struct wc_entry *e = &ctx->wc[ctx->wc_n++];
	memset(e, 0, sizeof(*e));
	e->key = ctx->wc_key; e->kind = kind; e->prec = prec;
	memcpy(e->d, d, nd * sizeof(int));
	e->last_use = ++ctx->wc_clock;
	e->scale = 1.f;
	return e;
}

/* Allocate the entry's NPU buffer; on failure the entry is dropped and NULL returned. */
static int buf_ensure(rk_ctx *ctx, rk_buf *b, size_t size, uint32_t flags);
static struct wc_entry *wc_alloc(rk_ctx *ctx, struct wc_entry *e, size_t bytes) {
	uint32_t dflags = MEM_FLAG_400 | RKNPU_MEM_CACHEABLE | RKNPU_MEM_NON_CONTIGUOUS;
	if (buf_ensure(ctx, &e->buf, bytes, dflags)) {
		rk_wcache_drop(ctx, e->key);
		return NULL;
	}
	ctx->wc_bytes += e->buf.size;
	return e;
}

void rk_close(rk_ctx *ctx) {
	if (!ctx) return;
	rk_wcache_clear(ctx);
	free(ctx->wc);
	buf_free(ctx, &ctx->regcmd);
	buf_free(ctx, &ctx->tasks);
	buf_free(ctx, &ctx->a);
	buf_free(ctx, &ctx->b);
	buf_free(ctx, &ctx->c);
	pool_destroy(ctx->pool);
	for (int i = 0; i < 3; i++) free(ctx->scr[i]);
	if (ctx->lock_fd >= 0) close(ctx->lock_fd);
	close(ctx->fd);
	free(ctx);
}

void rk_set_mode(rk_ctx *ctx, int mode) { ctx->mode = mode; }
void rk_set_threads(rk_ctx *ctx, int t) { ctx->threads = t > 0 ? t : 1; }
void rk_set_max_m_tile(rk_ctx *ctx, int m) { ctx->max_m = m >= 4 ? m & ~3 : 4; }
void rk_set_max_buf_mb(rk_ctx *ctx, int mb) { ctx->max_buf = (size_t)(mb > 1 ? mb : 1) << 20; }
void rk_get_stats(rk_ctx *ctx, rk_stats *out) { *out = ctx->stats; }

/* Serialize NPU submissions between processes using this library (the driver queues jobs
 * anyway; this keeps one process's task chains from interleaving with another's). */
static void npu_lock(rk_ctx *ctx) { if (ctx->lock_fd >= 0) flock(ctx->lock_fd, LOCK_EX); }
static void npu_unlock(rk_ctx *ctx) { if (ctx->lock_fd >= 0) flock(ctx->lock_fd, LOCK_UN); }

static void buf_free(rk_ctx *ctx, rk_buf *b);

void rk_wcache_clear(rk_ctx *ctx);
void rk_release(rk_ctx *ctx) {
	rk_wcache_clear(ctx);
	buf_free(ctx, &ctx->regcmd);
	buf_free(ctx, &ctx->tasks);
	buf_free(ctx, &ctx->a);
	buf_free(ctx, &ctx->b);
	buf_free(ctx, &ctx->c);
	for (int i = 0; i < 3; i++) { free(ctx->scr[i]); ctx->scr[i] = NULL; ctx->scr_n[i] = 0; }
}

size_t rk_buffer_bytes(rk_ctx *ctx) {
	return ctx->a.size + ctx->b.size + ctx->c.size + ctx->regcmd.size + ctx->tasks.size + ctx->wc_bytes;
}

int rk_query(rk_ctx *ctx, rk_info *info) {
	memset(info, 0, sizeof(*info));
	struct rknpu_action a;
	a.flags = RKNPU_GET_DRV_VERSION; a.value = 0;
	if (ioctl(ctx->fd, DRM_IOCTL_RKNPU_ACTION, &a) == 0) info->driver_version = (int)a.value;
	a.flags = RKNPU_GET_HW_VERSION; a.value = 0;
	if (ioctl(ctx->fd, DRM_IOCTL_RKNPU_ACTION, &a) == 0) info->hw_version = a.value;
	a.flags = RKNPU_GET_IOMMU_EN; a.value = 0;
	if (ioctl(ctx->fd, DRM_IOCTL_RKNPU_ACTION, &a) == 0) info->iommu = (int)a.value;
	a.flags = RKNPU_GET_FREQ; a.value = 0;
	if (ioctl(ctx->fd, DRM_IOCTL_RKNPU_ACTION, &a) == 0) info->freq_hz = a.value;
	return 0;
}

/* Grow-only buffer, cacheable, mapped on all NPU cores. */
static int buf_ensure(rk_ctx *ctx, rk_buf *b, size_t size, uint32_t flags) {
	if (b->ptr && b->size >= size && b->flags == flags) return 0;
	buf_free(ctx, b);
	size = (size + 4095) & ~(size_t)4095;
	if (size < 65536) size = 65536;
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
	if (flags & RKNPU_MEM_CACHEABLE) {
		/* flush lines left dirty by the kernel's page zeroing before the NPU ever writes here */
		struct rknpu_mem_sync sy = { .flags = RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE,
		                             .obj_addr = b->obj, .offset = 0, .size = size };
		ioctl(ctx->fd, DRM_IOCTL_RKNPU_MEM_SYNC, &sy);
	}
	return 0;
}

static void buf_sync(rk_ctx *ctx, rk_buf *b, size_t size, uint32_t dir) {
	if (!(b->flags & RKNPU_MEM_CACHEABLE)) return;
	double t0 = now_us();
	struct rknpu_mem_sync s = { .flags = dir, .obj_addr = b->obj, .offset = 0, .size = size };
	ioctl(ctx->fd, DRM_IOCTL_RKNPU_MEM_SYNC, &s);
	ctx->stats.sync_us += now_us() - t0;
}

/*
 * Per-precision constants, recovered from librknnrt command streams (fp16->fp32,
 * fp16->fp16, int8->int32) or probed (see tools/prec_probe.c).
 */
typedef struct {
	int in_prec, proc_prec, out_prec;   /* CNA input / MAC / DPU output precision codes */
	int in_bytes, out_bytes;
	int kgroup;                         /* kernels per weight group in the B layout */
	int qd_en;                          /* CORE_MISC_CFG bit 0 */
	uint32_t bs_ow;                     /* DPU_BS_OW_CFG */
	uint32_t cvt_scale, cvt_shift;      /* DPU_OUT_CVT_SCALE / SHIFT */
	uint32_t surf_add;                  /* DPU_SURFACE_ADD (>> 4) */
} dtype_t;

static const dtype_t DT_F16_F32 = { PREC_FP16, PREC_FP16, PREC_FP32, 2, 4, 16, 1, 0x36e, 1, 0, 4 };
static const dtype_t DT_F16_F16 = { PREC_FP16, PREC_FP16, PREC_FP16, 2, 2, 16, 0, 0x126, 0x10001, 0, 2 };
static const dtype_t DT_I8_I32  = { PREC_INT8, PREC_INT8, PREC_INT32, 1, 4, 32, 1, 0x7fc, 1, 0, 8 };
static const dtype_t DT_BF16_F32 = { PREC_BF16, PREC_BF16, PREC_FP32, 2, 4, 16, 1, 0x36e, 1, 0, 4 };

typedef struct {
	const dtype_t *dt;
	int m, k, n;            /* rows in this task, channels, kernels */
	int grains;             /* feature rows the CNA preloads */
	int dbanks, wbanks;     /* CBUF split */
	int reuse;              /* weights already resident from the previous task */
	uint32_t in_dma, w_dma, out_dma, next_dma;
	/* convolution geometry (all zero = matmul: width 1, height m, 1x1 kernel) */
	int in_w, in_h, out_w, out_h, kw, kh, sx, sy, pl, pt;
	int nc1hwc2;            /* 1: native NC1HWC2 feature/output layout (convolution) */
	int surf_in_h;          /* NC1HWC2: full input height (surface stride), 0 = in_h */
	int surf_out_hw;        /* NC1HWC2: full output plane (oh*ow) for surface strides, 0 = slice */
	int dw;                 /* depthwise convolution (CONV_MODE 3): one 2-D kernel per channel */
	int relu;               /* ReLU in the DPU's BS stage */
} task_desc;

/* Register stream for one row-major matmul task (precisions from t->dt). */
static void emit_task(uint64_t *o, const task_desc *t) {
	const dtype_t *dt = t->dt;
	const uint32_t m = t->m, k = t->k, n = t->n;
	const uint32_t kb = k * (uint32_t)dt->in_bytes;                   /* bytes per pixel (all channels) */
	const int conv = t->kw > 0;
	const uint32_t iw = conv ? (uint32_t)t->in_w : 1, ih = conv ? (uint32_t)t->in_h : m;
	const uint32_t ow = conv ? (uint32_t)t->out_w : 1, oh = conv ? (uint32_t)t->out_h : m;
	const uint32_t kw = conv ? (uint32_t)t->kw : 1, kh = conv ? (uint32_t)t->kh : 1;
	const uint32_t sx = conv ? (uint32_t)t->sx : 1, sy = conv ? (uint32_t)t->sy : 1;
	const uint32_t pl = conv ? (uint32_t)t->pl : 0, pt = conv ? (uint32_t)t->pt : 0;
	const uint32_t notch = n * (uint32_t)dt->out_bytes / 16 - 1;      /* 16-byte output atoms per row - 1 */
	const uint32_t cbuf = (t->reuse ? CBUF_WEIGHT_REUSE : 0) | ((uint32_t)t->wbanks << 4) | (uint32_t)t->dbanks;
	/* bit 29 (GROUP_LINE_OFF) with width 1 makes the CNA read A as a plain row-major matrix;
	 * convolutions use the native NC1HWC2 layout instead (16-byte channel atoms). */
	const int nc = t->nc1hwc2;
	const uint32_t conv1 = (nc ? 0u : (1u << 29)) | ((uint32_t)dt->proc_prec << 7) | ((uint32_t)dt->in_prec << 4) |
	                       (t->dw ? 3u : 0u);
	const uint32_t wk = t->dw ? 1u : n;                                  /* weight kernels */
	const uint32_t line_stride = nc ? 4 * iw : iw * kb / 16;
	const uint32_t sih = (nc && t->surf_in_h) ? (uint32_t)t->surf_in_h : ih;
	const uint32_t surf_stride = nc ? (uint32_t)((double)line_stride * ((double)sih / 4.0 - 1.0)) : 0;
	const uint32_t c_atoms = (kb + 15) / 16;                              /* 16-byte channel atoms per pixel */
	const uint32_t entries = nc ? (iw * 2 * c_atoms + 7) / 8 : (iw * kb + 63) / 64;
	const uint32_t dst_surf = nc ? (t->surf_out_hw ? (uint32_t)t->surf_out_hw : ow * oh) : 1;
	int i = 0;
#define R(blk, reg, val) (o[i++] = npuop((blk), (uint32_t)(val), (reg)))
	R(OP_REG_CNA, CNA_CBUF_CON0, cbuf);
	R(OP_REG_CNA, CNA_DCOMP_REGNUM, 0);
	R(OP_REG_CNA, CNA_DCOMP_CTRL, 0);
	R(OP_REG_CNA, CNA_CONV_CON1, conv1);
	R(OP_REG_DPU, DPU_S_POINTER, 0xE);
	R(OP_REG_CNA, CNA_CONV_CON1, conv1);
	R(OP_REG_CNA, CNA_CONV_CON2, ((uint32_t)t->grains & 0x3FF) << 4);
	R(OP_REG_CNA, CNA_CONV_CON3, ((sy & 7) << 3) | (sx & 7));
	R(OP_REG_CNA, CNA_DATA_SIZE0, ((iw & 0x7FF) << 16) | (ih & 0x7FF));
	R(OP_REG_CNA, CNA_DATA_SIZE1, ((k - 1) << 16) | k);
	R(OP_REG_CNA, CNA_DATA_SIZE2, ow & 0x7FF);
	R(OP_REG_CNA, CNA_DATA_SIZE3, (ow * oh) & 0x3FFFF);
	R(OP_REG_CNA, CNA_WEIGHT_SIZE0, kb * kw * kh * wk);
	R(OP_REG_CNA, CNA_WEIGHT_SIZE1, (kb * kw * kh) & 0x7FFFF);
	R(OP_REG_CNA, CNA_WEIGHT_SIZE2, ((kw & 0x1F) << 24) | ((kh & 0x1F) << 16) | (wk & 0x3FFF));
	R(OP_REG_CNA, CNA_CBUF_CON0, cbuf);
	R(OP_REG_CNA, CNA_CBUF_CON1, entries & 0x1FFF);
	R(OP_REG_CNA, CNA_CVT_CON0, (1 << 3) | (1 << 1) | 1);
	R(OP_REG_CNA, CNA_CVT_CON1, 1u << 16);
	R(OP_REG_CNA, CNA_CVT_CON2, 1u << 16);
	R(OP_REG_CNA, CNA_CVT_CON3, 1u << 16);
	R(OP_REG_CNA, CNA_CVT_CON4, 1u << 16);
	R(OP_REG_CNA, CNA_FC_CON0, 0);
	R(OP_REG_CNA, CNA_FC_CON1, 0);
	R(OP_REG_CNA, CNA_PAD_CON0, ((pl & 0xF) << 4) | (pt & 0xF));
	R(OP_REG_CNA, CNA_FEATURE_DATA_ADDR, t->in_dma);
	R(OP_REG_CNA, CNA_FC_CON2, 0);
	R(OP_REG_CNA, CNA_DMA_CON0, (15u << 16) | 15u);
	R(OP_REG_CNA, CNA_DMA_CON1, line_stride);
	R(OP_REG_CNA, CNA_DMA_CON2, surf_stride & 0xFFFFFFF);
	R(OP_REG_CNA, CNA_FC_DATA_SIZE0, ((iw & 0x7FF) << 16) | (ih & 0x7FF));
	R(OP_REG_CNA, CNA_FC_DATA_SIZE1, k & 0xFFFF);
	R(OP_REG_CNA, CNA_DCOMP_CTRL, 0);
	R(OP_REG_CNA, CNA_DCOMP_REGNUM, 0);
	R(OP_REG_CNA, CNA_DCOMP_ADDR0, t->w_dma);
	for (int r = 0; r < 16; r++) R(OP_REG_CNA, CNA_DCOMP_AMOUNT0 + 4 * r, 0);
	R(OP_REG_CNA, CNA_CVT_CON5, 0);
	R(OP_REG_CNA, CNA_PAD_CON1, 0);

	R(OP_REG_CORE, CORE_MISC_CFG, ((uint32_t)dt->proc_prec << 8) | (t->dw ? 2u : 0u) | (uint32_t)dt->qd_en);
	R(OP_REG_CORE, CORE_DATAOUT_SIZE_0, ((oh - 1) << 16) | (ow - 1));
	R(OP_REG_CORE, CORE_DATAOUT_SIZE_1, n - 1);
	R(OP_REG_CORE, CORE_CLIP_TRUNCATE, 0);
	R(OP_REG_CORE, CORE_3030, 0);

	R(OP_REG_DPU, DPU_FEATURE_MODE_CFG, (15u << 5) | (t->dw ? 3u << 3 : 0u) | (2u << 1));
	R(OP_REG_DPU, DPU_DATA_FORMAT, ((uint32_t)dt->out_prec << 29) | ((uint32_t)dt->proc_prec << 26) | (uint32_t)dt->proc_prec);
	R(OP_REG_DPU, DPU_OFFSET_PEND, 0);
	R(OP_REG_DPU, DPU_DST_BASE_ADD, t->out_dma);
	R(OP_REG_DPU, DPU_DST_SURF_STRIDE, (dst_surf & 0xFFFFFFF) << 4);  /* 1 = row-major output */
	R(OP_REG_DPU, DPU_DATA_CUBE_WIDTH, (ow - 1) & 0x1FFF);
	R(OP_REG_DPU, DPU_DATA_CUBE_HEIGHT, (oh - 1) & 0x1FFF);
	R(OP_REG_DPU, DPU_DATA_CUBE_NOTCH_ADDR, nc ? 0 : ((notch << 16) | notch));
	R(OP_REG_DPU, DPU_DATA_CUBE_CHANNEL, ((n - 1) << 16) | (n - 1));
	/* BS stage: everything bypassed, or (relu) only the ALU/multiplier bypassed so the ReLU runs */
	R(OP_REG_DPU, DPU_BS_CFG, t->relu ? ((1 << 4) | (1 << 1)) : ((1 << 6) | (1 << 4) | (1 << 1) | 1));
	R(OP_REG_DPU, DPU_BS_ALU_CFG, 0);
	R(OP_REG_DPU, DPU_BS_MUL_CFG, 0);
	R(OP_REG_DPU, DPU_BS_RELUX_CMP_VALUE, 0);
	R(OP_REG_DPU, DPU_BS_OW_CFG, dt->bs_ow);
	R(OP_REG_DPU, DPU_BS_OW_OP, 0);
	R(OP_REG_DPU, DPU_WDMA_SIZE_0, (n - 1) & 0x1FFF);
	R(OP_REG_DPU, DPU_WDMA_SIZE_1, (((oh - 1) & 0x1FFF) << 16) | ((ow - 1) & 0x1FFF));
	R(OP_REG_DPU, DPU_BN_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1);
	R(OP_REG_DPU, DPU_BN_ALU_CFG, 0);
	R(OP_REG_DPU, DPU_BN_MUL_CFG, 0);
	R(OP_REG_DPU, DPU_BN_RELUX_CMP_VALUE, 0);
	R(OP_REG_DPU, DPU_EW_CFG, (1 << 9) | (1 << 8) | (1 << 7) | (1 << 1) | 1);
	R(OP_REG_DPU, DPU_EW_CVT_OFFSET_VALUE, 0);
	R(OP_REG_DPU, DPU_EW_CVT_SCALE_VALUE, 1);
	R(OP_REG_DPU, DPU_EW_RELUX_CMP_VALUE, 0);
	R(OP_REG_DPU, DPU_OUT_CVT_OFFSET, 0);
	R(OP_REG_DPU, DPU_OUT_CVT_SCALE, dt->cvt_scale);
	R(OP_REG_DPU, DPU_OUT_CVT_SHIFT, dt->cvt_shift);
	for (int r = 0; r < 8; r++) R(OP_REG_DPU, DPU_EW_OP_VALUE_0 + 4 * r, 0);
	R(OP_REG_DPU, DPU_SURFACE_ADD, ((dst_surf * dt->surf_add) & 0xFFFFFFF) << 4);
	R(OP_REG_DPU, DPU_40C4, 0);
	for (int r = 0; r < 12; r++) R(OP_REG_DPU, DPU_LUT_ACCESS_CFG + 4 * r, 0);
	/* i == 108: PC tail, chaining to the next task's register stream if any */
	if (t->next_dma) {
		R(OP_REG_PC, PC_BASE_ADDRESS, t->next_dma);
		R(OP_REG_PC, PC_REGISTER_AMOUNTS, PC_AMOUNT);
	} else {
		o[i++] = 0;
		R(OP_REG_PC, PC_REGISTER_AMOUNTS, 0);
	}
	o[i++] = npuop(OP_40, 0, 0);
	o[i++] = npuop(OP_ENABLE, PC_ENABLE_DPU | PC_ENABLE_CNA | PC_ENABLE, PC_OPERATION_ENABLE);
#undef R
}

static int submit(rk_ctx *ctx, int start, int count, int cores) {
	struct rknpu_submit s;
	memset(&s, 0, sizeof(s));
	s.flags = RKNPU_JOB_PC | RKNPU_JOB_PINGPONG;
	s.timeout = ctx->timeout_ms;
	s.task_obj_addr = ctx->tasks.obj;
	s.fence_fd = -1;
	if (cores <= 1) {
		/* same shape librknnrt uses: auto core, every slot naming the range, task_number = sum */
		s.core_mask = 0;
		s.task_start = start;
		s.task_number = count * 3;
		for (int c = 0; c < 3; c++) {
			s.subcore_task[c].task_start = start;
			s.subcore_task[c].task_number = count;
		}
	} else {
		/* three cores: the driver reads subcore_task[2..4] when three cores are in the mask */
		s.core_mask = 0x7;
		s.task_start = start;
		s.task_number = count;
		int base = start;
		for (int c = 0; c < 3; c++) {
			int n = count / 3 + (c < count % 3);
			s.subcore_task[c + 2].task_start = base;
			s.subcore_task[c + 2].task_number = n;
			base += n;
		}
	}
	double t0 = now_us();
	int r = ioctl(ctx->fd, DRM_IOCTL_RKNPU_SUBMIT, &s);
	ctx->stats.submit_us += now_us() - t0;
	ctx->stats.submits++;
	if (r < 0) {
		set_err(ctx, "SUBMIT(start=%d, n=%d, cores=%d) failed: %s", start, count, cores, strerror(errno));
		return -1;
	}
	ctx->stats.hw_us += s.hw_elapse_time / 1000.0;
	return 0;
}

/*
 * Worker pool for packing/unpacking. Workers are pinned to the big cores: an unpinned
 * pool often lands on the Cortex-A55 cluster, which made packing ~10x slower. Workers
 * sleep on a condition variable between jobs (no spinning, so they don't steal cycles
 * from PyTorch's own threads), work is handed out dynamically so the calling thread can
 * help wherever the scheduler put it, and the caller's affinity is never changed.
 */
typedef void (*range_fn)(void *arg, int begin, int end);

struct pool {
	int nworkers;
	pthread_t th[16];
	int cpu[16];
	pthread_mutex_t mu;
	pthread_cond_t start_cv, done_cv;
	unsigned gen;
	int active;          /* workers still inside the current job */
	int quit;
	range_fn fn;
	void *arg;
	int total, grain;
	int next;            /* next unclaimed item (atomic) */
};

typedef struct { struct pool *p; int idx; } worker_arg;

static void pool_drain(struct pool *p) {
	for (;;) {
		int b = __atomic_fetch_add(&p->next, p->grain, __ATOMIC_RELAXED);
		if (b >= p->total) break;
		int e = b + p->grain < p->total ? b + p->grain : p->total;
		p->fn(p->arg, b, e);
	}
}

static void *worker_main(void *v) {
	worker_arg *wa = v;
	struct pool *p = wa->p;
	if (p->cpu[wa->idx] >= 0) {
		cpu_set_t set;
		CPU_ZERO(&set);
		CPU_SET(p->cpu[wa->idx], &set);
		sched_setaffinity(0, sizeof(set), &set);
	}
	free(wa);
	unsigned seen = 0;
	pthread_mutex_lock(&p->mu);
	for (;;) {
		while (p->gen == seen && !p->quit) pthread_cond_wait(&p->start_cv, &p->mu);
		if (p->quit) break;
		seen = p->gen;
		pthread_mutex_unlock(&p->mu);
		pool_drain(p);
		pthread_mutex_lock(&p->mu);
		if (--p->active == 0) pthread_cond_signal(&p->done_cv);
	}
	pthread_mutex_unlock(&p->mu);
	return NULL;
}

static struct pool *pool_create(int nworkers) {
	struct pool *p = calloc(1, sizeof(*p));
	if (nworkers > 16) nworkers = 16;
	pthread_mutex_init(&p->mu, NULL);
	pthread_cond_init(&p->start_cv, NULL);
	pthread_cond_init(&p->done_cv, NULL);
	/* big cores = highest cpu_capacity; leave the first one to the caller / PyTorch */
	int caps[64], ncpu = 0, maxcap = 0, big[64], nbig = 0;
	for (int c = 0; c < 64; c++) {
		char path[96];
		snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpu_capacity", c);
		FILE *f = fopen(path, "r");
		if (!f) break;
		if (fscanf(f, "%d", &caps[c]) != 1) caps[c] = 0;
		fclose(f);
		if (caps[c] > maxcap) maxcap = caps[c];
		ncpu = c + 1;
	}
	for (int c = 0; c < ncpu; c++) if (caps[c] == maxcap) big[nbig++] = c;
	int pin = !env_int("RKNPU_NO_PIN", 0) && nbig > 0 && nbig < ncpu;
	int first = nbig > nworkers ? nbig - nworkers : 0;
	p->nworkers = nworkers;
	for (int i = 0; i < nworkers; i++) {
		p->cpu[i] = pin ? big[first + i % (nbig - first)] : -1;
		worker_arg *wa = malloc(sizeof(*wa));
		wa->p = p;
		wa->idx = i;
		pthread_create(&p->th[i], NULL, worker_main, wa);
	}
	return p;
}

static void pool_destroy(struct pool *p) {
	if (!p) return;
	pthread_mutex_lock(&p->mu);
	p->quit = 1;
	pthread_cond_broadcast(&p->start_cv);
	pthread_mutex_unlock(&p->mu);
	for (int i = 0; i < p->nworkers; i++) pthread_join(p->th[i], NULL);
	free(p);
}

/* Run fn over [0, total) in chunks of `grain` on the workers plus the calling thread. */
static void parallel_for(rk_ctx *ctx, int total, int grain, range_fn fn, void *arg) {
	if (total <= 0) return;
	if (grain < 1) grain = 1;
	if (ctx->threads <= 1 || total <= grain) { fn(arg, 0, total); return; }
	if (!ctx->pool) ctx->pool = pool_create(ctx->threads);
	struct pool *p = ctx->pool;
	pthread_mutex_lock(&p->mu);
	p->fn = fn;
	p->arg = arg;
	p->total = total;
	p->grain = grain;
	p->next = 0;
	p->active = p->nworkers;
	p->gen++;
	pthread_cond_broadcast(&p->start_cv);
	pthread_mutex_unlock(&p->mu);
	pool_drain(p);
	pthread_mutex_lock(&p->mu);
	while (p->active > 0) pthread_cond_wait(&p->done_cv, &p->mu);
	pthread_mutex_unlock(&p->mu);
}

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
static inline int round_up(int a, int b) { return ceil_div(a, b) * b; }

/* ------------------------------------------------------------------------------------
 * Element conversion
 * ---------------------------------------------------------------------------------- */
enum { CV_F16, CV_BF16, CV_I8 };

static inline uint16_t f32_to_bf16(float x) {
	uint32_t u;
	memcpy(&u, &x, 4);
	if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);  /* NaN stays NaN */
	u += 0x7fffu + ((u >> 16) & 1u);                                          /* round to nearest even */
	return (uint16_t)(u >> 16);
}

static inline int8_t f32_to_i8(float x) {
	x = x > 127.f ? 127.f : (x < -127.f ? -127.f : x);
	return (int8_t)(int)(x + __builtin_copysignf(0.5f, x));   /* truncation after +-0.5 = round */
}

typedef struct {
	const dtype_t *dt;
	int cv;                               /* CV_* input conversion */
	const float *A, *B;
	int64_t sa_m, sa_k, sb_k, sb_n;
	int M, N, K, Mp, Kc, Nc, nk, nn, nblk;
	void *ap, *bp, *cp;
	size_t a_chunk, b_tile, c_tile;       /* elements */
	float a_scale, b_scale;               /* uniform input scales (fp16/bf16) */
	const float *a_row, *b_col;           /* per-row / per-column scales (int8) */
	const float *b_inv;                   /* 1 / b_col */
	float *C;
	int64_t ldc;
	int accumulate;
	float out_mul;                        /* applied to every output (undo input scales) */
	float *amax_part;                     /* scratch for max-abs reductions */
	int part_stride;                      /* amax_part elements per batch */
	int batch;                            /* independent GEMMs sharing one shape */
	int64_t sa_b, sb_b, sc_b;             /* batch strides of A, B, C (elements) */
	size_t a_bat, b_bat, c_bat;           /* NPU buffer elements per batch */
	const float *bias;                    /* epilogue: + bias[n] */
	int relu;                             /* epilogue: max(., 0) */
	int relu_dpu;                         /* the ReLU already ran in the DPU */
	struct wc_entry *wc;                  /* cached packed B (or NULL) */
} job_t;

/* write one converted element of A row m / B column n */
#define CONV_A(J, x, m) ((J)->cv == CV_I8 ? (x) * (J)->a_row[m] : (x) * (J)->a_scale)
#define CONV_B(J, x, n) ((J)->cv == CV_I8 ? (x) * (J)->b_col[n] : (x) * (J)->b_scale)

static inline void store_el(int cv, void *base, size_t idx, float v) {
	if (cv == CV_F16) ((f16 *)base)[idx] = (f16)v;
	else if (cv == CV_BF16) ((uint16_t *)base)[idx] = f32_to_bf16(v);
	else ((int8_t *)base)[idx] = f32_to_i8(v);
}

/* A: row-major [Mp][Kc] per K chunk; items are (K chunk, 16-row block) */
static void pack_a_range(void *v, int b, int e) {
	job_t *J = v;
	const int Kc = J->Kc, K = J->K, M = J->M, cv = J->cv;
	for (int it = b; it < e; it++) {
		int j = it / J->nblk, bk = it % J->nblk;
		int r0 = bk * 16, r1 = r0 + 16 < J->Mp ? r0 + 16 : J->Mp;
		int k0 = j * Kc;
		size_t base = (size_t)j * J->a_chunk;
		int kend = K - k0 < Kc ? K - k0 : Kc;
		if (J->sa_m == 1 && J->sa_k != 1) {
			/* transposed source: walk k-major so reads are contiguous along m */
			for (int k = 0; k < Kc; k++) {
				const float *s = J->A + (int64_t)(k0 + k) * J->sa_k;
				for (int m = r0; m < r1; m++) {
					float x = (m < M && k < kend) ? CONV_A(J, s[m], m) : 0.f;
					store_el(cv, J->ap, base + (size_t)m * Kc + k, x);
				}
			}
			continue;
		}
		for (int m = r0; m < r1; m++) {
			size_t d = base + (size_t)m * Kc;
			if (m >= M) { for (int k = 0; k < Kc; k++) store_el(cv, J->ap, d + k, 0.f); continue; }
			const float *s = J->A + (int64_t)m * J->sa_m + (int64_t)k0 * J->sa_k;
			if (cv == CV_F16 && J->sa_k == 1) {
				f16 *dd = (f16 *)J->ap + d;
				const float sc = J->a_scale;
				for (int k = 0; k < kend; k++) dd[k] = (f16)(s[k] * sc);
				for (int k = kend; k < Kc; k++) dd[k] = (f16)0;
			} else if (cv == CV_BF16 && J->sa_k == 1) {
				uint16_t *dd = (uint16_t *)J->ap + d;
				for (int k = 0; k < kend; k++) dd[k] = f32_to_bf16(s[k]);
				for (int k = kend; k < Kc; k++) dd[k] = 0;
			} else if (cv == CV_I8 && J->sa_k == 1) {
				int8_t *dd = (int8_t *)J->ap + d;
				const float sc = J->a_row[m];
				for (int k = 0; k < kend; k++) dd[k] = f32_to_i8(s[k] * sc);
				for (int k = kend; k < Kc; k++) dd[k] = 0;
			} else {
				for (int k = 0; k < kend; k++) store_el(cv, J->ap, d + k, CONV_A(J, s[(int64_t)k * J->sa_k], m));
				for (int k = kend; k < Kc; k++) store_el(cv, J->ap, d + k, 0.f);
			}
		}
	}
}

/* B: kernel groups [Nc/G][Kc/32][G][32] per (K chunk, N chunk); items are G-kernel groups */
static void pack_b_range(void *v, int b, int e) {
	job_t *J = v;
	const int Kc = J->Kc, Nc = J->Nc, K = J->K, N = J->N, G = J->dt->kgroup, cv = J->cv;
	const int64_t sb_k = J->sb_k, sb_n = J->sb_n;
	for (int it = b; it < e; it++) {
		int jc = it / (Nc / G), ng = it % (Nc / G);
		int j = jc / J->nn, c = jc % J->nn;
		size_t dst = (size_t)jc * J->b_tile + (size_t)ng * G * Kc;
		for (int cb = 0; cb < Kc / 32; cb++) {
			for (int nr = 0; nr < G; nr++) {
				int gn = c * Nc + ng * G + nr;
				size_t d = dst + (size_t)cb * 32 * G + (size_t)nr * 32;
				int k0 = j * Kc + cb * 32;
				for (int t = 0; t < 32; t++) {
					int gk = k0 + t;
					float x = (gn < N && gk < K) ? CONV_B(J, J->B[(int64_t)gk * sb_k + (int64_t)gn * sb_n], gn) : 0.f;
					store_el(cv, J->bp, d + t, x);
				}
			}
		}
	}
}

/* Typed fast B packers: layout [Nc/G][Kc/32][G][32]; both source orders handled with
 * contiguous reads (row-major B: 32 k-rows x G consecutive n; column-major: 32 consecutive k). */
#define DEFINE_PACK_B(NAME, T, G, CONV)                                                          \
static void NAME(void *v, int b, int e) {                                                        \
	job_t *J = v;                                                                                \
	const int Kc = J->Kc, Nc = J->Nc, K = J->K, N = J->N;                                        \
	for (int it = b; it < e; it++) {                                                             \
		int jc = it / (Nc / G), ng = it % (Nc / G);                                              \
		int j = jc / J->nn, c = jc % J->nn;                                                      \
		T *dst = (T *)J->bp + (size_t)jc * J->b_tile + (size_t)ng * G * Kc;                      \
		int nb = c * Nc + ng * G;                                                                \
		if (J->sb_n == 1 && J->sb_k != 1) {                                                      \
			for (int cb = 0; cb < Kc / 32; cb++)                                                 \
				for (int kr = 0; kr < 32; kr++) {                                                \
					int gk = j * Kc + cb * 32 + kr;                                              \
					const float *s = J->B + (int64_t)gk * J->sb_k + nb;                          \
					for (int nr = 0; nr < G; nr++) {                                             \
						int gn = nb + nr;                                                        \
						float x = (gk < K && gn < N) ? s[nr] : 0.f;                              \
						dst[cb * 32 * G + nr * 32 + kr] = CONV(x, gn);                           \
					}                                                                            \
				}                                                                                \
		} else {                                                                                 \
			for (int nr = 0; nr < G; nr++) {                                                     \
				int gn = nb + nr;                                                                \
				const float *s = J->B + (int64_t)gn * J->sb_n;                                   \
				for (int cb = 0; cb < Kc / 32; cb++) {                                           \
					T *d = dst + cb * 32 * G + nr * 32;                                          \
					int k0 = j * Kc + cb * 32;                                                   \
					if (gn < N && k0 + 32 <= K) {                                                \
						if (J->sb_k == 1) { for (int t = 0; t < 32; t++) d[t] = CONV(s[k0 + t], gn); } \
						else { for (int t = 0; t < 32; t++) d[t] = CONV(s[(int64_t)(k0 + t) * J->sb_k], gn); } \
					} else {                                                                     \
						for (int t = 0; t < 32; t++) {                                           \
							int gk = k0 + t;                                                     \
							float x = (gn < N && gk < K) ? s[(int64_t)gk * J->sb_k] : 0.f;       \
							d[t] = CONV(x, gn);                                                  \
						}                                                                        \
					}                                                                            \
				}                                                                                \
			}                                                                                    \
		}                                                                                        \
	}                                                                                            \
}
#define CONV_F16(x, n)  ((f16)((x) * J->b_scale))
#define CONV_BF16(x, n) (f32_to_bf16((x) * J->b_scale))
#define CONV_I8(x, n)   (f32_to_i8((x) * J->b_col[n]))
DEFINE_PACK_B(pack_b_f16_range, f16, 16, CONV_F16)
DEFINE_PACK_B(pack_b_bf16_range, uint16_t, 16, CONV_BF16)
DEFINE_PACK_B(pack_b_i8_range, int8_t, 32, CONV_I8)

/* output epilogue on one finished row of C */
static inline void epi_row(float *crow, int N, const float *bias, int relu) {
	if (bias && relu) { for (int n = 0; n < N; n++) { float x = crow[n] + bias[n]; crow[n] = x > 0.f ? x : 0.f; } }
	else if (bias) { for (int n = 0; n < N; n++) crow[n] += bias[n]; }
	else if (relu) { for (int n = 0; n < N; n++) crow[n] = crow[n] > 0.f ? crow[n] : 0.f; }
}

/* C: row-major [Mp][Nc] tiles -> C, reducing K chunks and undoing input scales */
static void unpack_core(void *v, int b, int e);
static void unpack_range(void *v, int b, int e) {
	job_t *J = v;
	unpack_core(v, b, e);
	int relu = J->relu && !J->relu_dpu;
	if (J->bias || relu)
		for (int m = b; m < e; m++) epi_row(J->C + (int64_t)m * J->ldc, J->N, J->bias, relu);
}
static void unpack_core(void *v, int b, int e) {
	job_t *J = v;
	const int ob = J->dt->out_prec;
	if (ob == PREC_INT32 && J->nk == 1) {
		for (int m = b; m < e; m++) {
			float *crow = J->C + (int64_t)m * J->ldc;
			const float ra = J->out_mul / J->a_row[m];
			for (int c = 0; c < J->nn; c++) {
				int n0 = c * J->Nc;
				int ncols = J->N - n0 < J->Nc ? J->N - n0 : J->Nc;
				const int32_t *src = (const int32_t *)J->cp + (size_t)c * J->c_tile + (size_t)m * J->Nc;
				const float *bi = J->b_inv + n0;
				if (J->accumulate) { for (int n = 0; n < ncols; n++) crow[n0 + n] += (float)src[n] * ra * bi[n]; }
				else { for (int n = 0; n < ncols; n++) crow[n0 + n] = (float)src[n] * ra * bi[n]; }
			}
		}
		return;
	}
	if (ob == PREC_FP32 && J->nk == 1) {
		/* fast path: one K chunk, fp32 tiles */
		const float mul = J->out_mul;
		for (int m = b; m < e; m++) {
			float *crow = J->C + (int64_t)m * J->ldc;
			for (int c = 0; c < J->nn; c++) {
				int n0 = c * J->Nc;
				int ncols = J->N - n0 < J->Nc ? J->N - n0 : J->Nc;
				const float *src = (const float *)J->cp + (size_t)c * J->c_tile + (size_t)m * J->Nc;
				if (J->accumulate) { for (int n = 0; n < ncols; n++) crow[n0 + n] += src[n] * mul; }
				else if (mul == 1.f) memcpy(crow + n0, src, (size_t)ncols * 4);
				else { for (int n = 0; n < ncols; n++) crow[n0 + n] = src[n] * mul; }
			}
		}
		return;
	}
	for (int m = b; m < e; m++) {
		float *crow = J->C + (int64_t)m * J->ldc;
		const float rs = J->cv == CV_I8 ? J->a_row[m] : 1.f;
		for (int c = 0; c < J->nn; c++) {
			int n0 = c * J->Nc;
			int ncols = J->N - n0 < J->Nc ? J->N - n0 : J->Nc;
			for (int n = 0; n < ncols; n++) {
				size_t off = (size_t)m * J->Nc + n;
				float acc;
				if (ob == PREC_INT32) {
					int64_t s = 0;
					for (int jj = 0; jj < J->nk; jj++) s += ((int32_t *)J->cp)[(size_t)(jj * J->nn + c) * J->c_tile + off];
					/* the int8 scales are multipliers into int8 range: divide them back out */
					acc = (float)s / (rs * J->b_col[n0 + n]);
				} else if (ob == PREC_FP16) {
					acc = 0.f;
					for (int jj = 0; jj < J->nk; jj++) acc += (float)((f16 *)J->cp)[(size_t)(jj * J->nn + c) * J->c_tile + off];
				} else {
					acc = 0.f;
					for (int jj = 0; jj < J->nk; jj++) acc += ((float *)J->cp)[(size_t)(jj * J->nn + c) * J->c_tile + off];
				}
				acc *= J->out_mul;
				if (J->accumulate) crow[n0 + n] += acc; else crow[n0 + n] = acc;
			}
		}
	}
}

/* max-abs scans on the IEEE bit patterns (sign cleared): integer max vectorizes cleanly
 * and orders non-negative floats correctly (NaN sorts above inf, handled by pow2_scale). */
static inline uint32_t absbits(float x) { uint32_t u; memcpy(&u, &x, 4); return u & 0x7fffffffu; }
static inline float bitsf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

/* max |A(m,:)| per row into amax_part[m]; items are 64-row blocks */
static void amax_a_rows(void *v, int b, int e) {
	job_t *J = v;
	uint32_t *mx = (uint32_t *)J->amax_part;
	for (int it = b; it < e; it++) {
		int m0 = it * 64, m1 = m0 + 64 < J->M ? m0 + 64 : J->M;
		if (J->sa_k == 1) {
			for (int m = m0; m < m1; m++) {
				const uint32_t *s = (const uint32_t *)(J->A + (int64_t)m * J->sa_m);
				uint32_t x = 0;
				for (int k = 0; k < J->K; k++) { uint32_t a = s[k] & 0x7fffffffu; x = a > x ? a : x; }
				mx[m] = x;
			}
		} else {
			for (int m = m0; m < m1; m++) mx[m] = 0;
			for (int k = 0; k < J->K; k++) {
				const float *s = J->A + (int64_t)k * J->sa_k;
				if (J->sa_m == 1) {
					const uint32_t *su = (const uint32_t *)s;
					for (int m = m0; m < m1; m++) { uint32_t a = su[m] & 0x7fffffffu; mx[m] = a > mx[m] ? a : mx[m]; }
				} else {
					for (int m = m0; m < m1; m++) { uint32_t a = absbits(s[(int64_t)m * J->sa_m]); mx[m] = a > mx[m] ? a : mx[m]; }
				}
			}
		}
		for (int m = m0; m < m1; m++) J->amax_part[m] = bitsf(mx[m]);
	}
}
/* max |B(:,n)| per column into amax_part[n]; items are 64-column blocks */
static void amax_b_cols(void *v, int b, int e) {
	job_t *J = v;
	uint32_t *mx = (uint32_t *)J->amax_part;
	for (int it = b; it < e; it++) {
		int n0 = it * 64, n1 = n0 + 64 < J->N ? n0 + 64 : J->N;
		if (J->sb_k == 1) {
			for (int n = n0; n < n1; n++) {
				const uint32_t *s = (const uint32_t *)(J->B + (int64_t)n * J->sb_n);
				uint32_t x = 0;
				for (int k = 0; k < J->K; k++) { uint32_t a = s[k] & 0x7fffffffu; x = a > x ? a : x; }
				mx[n] = x;
			}
		} else {
			for (int n = n0; n < n1; n++) mx[n] = 0;
			for (int k = 0; k < J->K; k++) {
				const float *s = J->B + (int64_t)k * J->sb_k;
				if (J->sb_n == 1) {
					const uint32_t *su = (const uint32_t *)s;
					for (int n = n0; n < n1; n++) { uint32_t a = su[n] & 0x7fffffffu; mx[n] = a > mx[n] ? a : mx[n]; }
				} else {
					for (int n = n0; n < n1; n++) { uint32_t a = absbits(s[(int64_t)n * J->sb_n]); mx[n] = a > mx[n] ? a : mx[n]; }
				}
			}
		}
		for (int n = n0; n < n1; n++) J->amax_part[n] = bitsf(mx[n]);
	}
}

/* power of two that puts `amax` near 2^14, well inside fp16 range on both ends */
static float pow2_scale(float amax) {
	if (!(amax > 0.f) || amax != amax || amax > 3e38f) return 1.f;
	int e;
	frexpf(amax, &e);            /* amax = f * 2^e, f in [0.5, 1) */
	int sh = 14 - e;
	if (sh > 100) sh = 100;
	if (sh < -100) sh = -100;
	return ldexpf(1.f, sh);
}

/* ------------------------------------------------------------------------------------
 * Batching: every range function sees a per-batch view of the job
 * ---------------------------------------------------------------------------------- */
static void job_at_batch(const job_t *J, job_t *L, int bb) {
	*L = *J;
	if (bb == 0) return;
	int ib = J->dt ? J->dt->in_bytes : 2, ob = J->dt ? J->dt->out_bytes : 4;
	if (L->A) L->A += bb * J->sa_b;
	if (L->B) L->B += bb * J->sb_b;
	if (L->C) L->C += bb * J->sc_b;
	if (L->ap) L->ap = (char *)L->ap + (size_t)bb * J->a_bat * ib;
	if (L->bp) L->bp = (char *)L->bp + (size_t)bb * J->b_bat * ib;
	if (L->cp) L->cp = (char *)L->cp + (size_t)bb * J->c_bat * ob;
	if (L->a_row) L->a_row += (size_t)bb * J->M;
	if (L->b_col) L->b_col += (size_t)bb * J->N;
	if (L->b_inv) L->b_inv += (size_t)bb * J->N;
	if (L->amax_part) L->amax_part += (size_t)bb * J->part_stride;
}

typedef struct { job_t *J; range_fn fn; int per; } bwrap;

static void batched_range(void *v, int b, int e) {
	bwrap *W = v;
	for (int it = b; it < e;) {
		int bb = it / W->per, lo = it % W->per;
		int hi = (W->per - lo) < (e - it) ? W->per : lo + (e - it);
		job_t L;
		job_at_batch(W->J, &L, bb);
		W->fn(&L, lo, hi);
		it += hi - lo;
	}
}

static void pfor_batched(rk_ctx *ctx, job_t *J, int per, int grain, range_fn fn) {
	if (J->batch <= 1) { parallel_for(ctx, per, grain, fn, J); return; }
	bwrap W = { J, fn, per };
	parallel_for(ctx, per * J->batch, grain, batched_range, &W);
}

/* ------------------------------------------------------------------------------------
 * Engine: one tiled, chained, multi-core NPU matmul (or a batch of same-shape ones)
 * ---------------------------------------------------------------------------------- */
static int gemm_engine(rk_ctx *ctx, job_t *J) {
	const dtype_t *dt = J->dt;
	const int ib = dt->in_bytes, ob = dt->out_bytes, G = dt->kgroup;
	const int M = J->M, N = J->N, K = J->K;

	int Kp = round_up(K, 32);
	int nk = ceil_div(Kp, ctx->max_k);
	int Kc = round_up(ceil_div(Kp, nk), 32);
	int Np = round_up(N, G);
	int nn = ceil_div(Np, ctx->max_n);
	int Nc = round_up(ceil_div(Np, nn), G);
	int Mp = round_up(M, 4);
	/* weight banks: at least one kernel group, and (like librknnrt) up to 4 banks or
	 * whatever holds all weights; leaving a single bank for streamed weights hangs the CNA */
	int wbanks_min = ceil_div(G * Kc * ib, CBUF_BANK_SIZE);
	int wall = ceil_div(Kc * ib * Nc, CBUF_BANK_SIZE);
	int wfloor = env_int("RKNPU_MIN_WBANKS", 4);
	if (wall < wfloor) wfloor = wall;
	if (wbanks_min < wfloor) wbanks_min = wfloor;
	int dbanks_max = CBUF_BANKS - wbanks_min;
	if (dbanks_max < 1) { set_err(ctx, "K chunk %d too large for CBUF", Kc); return -1; }
	/* rows per task: (rows + 1) must fit in the data banks so the CNA can preload rows + 1 */
	int Mt = (int)((dbanks_max * (long)CBUF_BANK_SIZE) / (Kc * ib)) - 1;
	Mt &= ~3;
	if (Mt > ctx->max_m) Mt = ctx->max_m & ~3;
	if (Mt > Mp) Mt = Mp;
	if (Mt < 4) { set_err(ctx, "K chunk %d leaves no room for 4 rows", Kc); return -1; }
	int nm = ceil_div(Mp, Mt);
	Mt = round_up(ceil_div(Mp, nm), 4);   /* rebalance */
	Mp = Mt * nm;
	int dbanks = ceil_div((Mt + 1) * Kc * ib, CBUF_BANK_SIZE);
	int wbanks = CBUF_BANKS - dbanks;
	int weights_fit = (long)Kc * ib * Nc <= (long)wbanks * CBUF_BANK_SIZE;
	int T = nm * nk * nn;                     /* tasks per batch element */
	int nb = J->batch > 1 ? J->batch : 1;
	int TT = T * nb;
	ctx->stats.m_tile = Mt; ctx->stats.k_tile = Kc; ctx->stats.n_tile = Nc;
	ctx->stats.tasks += TT;

	size_t a_chunk = (size_t)Mp * Kc, b_tile = (size_t)Nc * Kc, c_tile = (size_t)Mp * Nc;
	J->a_bat = a_chunk * nk; J->b_bat = b_tile * nk * nn; J->c_bat = c_tile * nk * nn;
	size_t a_bytes = J->a_bat * nb * ib, b_bytes = J->b_bat * nb * ib, c_bytes = J->c_bat * nb * ob;
	uint32_t dflags = MEM_FLAG_400 | RKNPU_MEM_CACHEABLE | RKNPU_MEM_NON_CONTIGUOUS;
	struct wc_entry *wc = J->wc;
	if (wc && wc->packed && (wc->layout[0] != Kc || wc->layout[1] != Nc || wc->layout[2] != nk || wc->layout[3] != nn))
		wc = NULL;   /* packed for another tiling: don't use */
	if (wc && !wc->packed) wc = wc_alloc(ctx, wc, b_bytes);
	if (buf_ensure(ctx, &ctx->a, a_bytes, dflags) ||
	    (!wc && buf_ensure(ctx, &ctx->b, b_bytes, dflags)) ||
	    buf_ensure(ctx, &ctx->c, c_bytes, dflags) ||
	    buf_ensure(ctx, &ctx->regcmd, (size_t)TT * REGCMD_SLOTS * 8, dflags) ||
	    buf_ensure(ctx, &ctx->tasks, (size_t)TT * sizeof(struct rknpu_task), dflags | RKNPU_MEM_KERNEL_MAPPING))
		return -1;

	J->Mp = Mp; J->Kc = Kc; J->Nc = Nc; J->nk = nk; J->nn = nn;
	J->ap = ctx->a.ptr; J->bp = wc ? wc->buf.ptr : ctx->b.ptr; J->cp = ctx->c.ptr;
	const uint64_t b_dma = wc ? wc->buf.dma : ctx->b.dma;
	const int pack_b = J->B && !(wc && wc->packed);
	/* ReLU in the DPU when nothing is added after the NPU's own output */
	J->relu_dpu = J->relu && !J->bias && nk == 1 && !J->accumulate && dt->out_prec != PREC_FP16;
	J->a_chunk = a_chunk; J->b_tile = b_tile; J->c_tile = c_tile;
	J->nblk = ceil_div(Mp, 16);

	double t0 = now_us();
	if (J->A) pfor_batched(ctx, J, nk * J->nblk, 4, pack_a_range);
	if (pack_b) {
		range_fn pb = pack_b_range;
		if (J->cv == CV_F16 && G == 16) pb = pack_b_f16_range;
		else if (J->cv == CV_BF16 && G == 16) pb = pack_b_bf16_range;
		else if (J->cv == CV_I8 && G == 32) pb = pack_b_i8_range;
		pfor_batched(ctx, J, nk * nn * (Nc / G), 1, pb);
	}
	ctx->stats.pack_us += now_us() - t0;
	buf_sync(ctx, &ctx->a, a_bytes, RKNPU_MEM_SYNC_TO_DEVICE);
	if (pack_b) {
		buf_sync(ctx, wc ? &wc->buf : &ctx->b, b_bytes, RKNPU_MEM_SYNC_TO_DEVICE);
		if (wc) {
			wc->packed = 1;
			wc->layout[0] = Kc; wc->layout[1] = Nc; wc->layout[2] = nk; wc->layout[3] = nn;
		}
	} else if (wc) {
		ctx->stats.cache_hits++;
	}

	/* register streams: task order (j, c, i) so consecutive tasks share weights */
	uint64_t *rc = ctx->regcmd.ptr;
	struct rknpu_task *tk = ctx->tasks.ptr;
	int cores = ctx->mode == 2 ? 3 : 1;
	int per_submit = ctx->mode == 0 ? 1 : PC_TASK_LIMIT;
	for (int t = 0; t < TT; t++) {
		int bb = t / T, tl = t % T;
		int i = tl % nm, jc = tl / nm, j = jc / nn;
		int sub_start = (t / per_submit) * per_submit;
		int sub_count = TT - sub_start < per_submit ? TT - sub_start : per_submit;
		int first = 0, last = 0;
		if (ctx->mode == 0) first = last = 1;
		else if (cores == 1 || sub_count < 3) { first = t == sub_start; last = t == sub_start + sub_count - 1; }
		else {
			int base = sub_start;
			for (int cc = 0; cc < 3; cc++) {
				int n = sub_count / 3 + (cc < sub_count % 3);
				if (t == base) first = 1;
				if (t == base + n - 1) last = 1;
				base += n;
			}
		}
		task_desc d = {
			.dt = dt, .m = Mt, .k = Kc, .n = Nc, .grains = Mt + 1, .dbanks = dbanks, .wbanks = wbanks,
			.reuse = ctx->reuse && weights_fit && !first && i > 0,
			.in_dma = (uint32_t)(ctx->a.dma + ((size_t)bb * J->a_bat + (size_t)j * a_chunk + (size_t)i * Mt * Kc) * ib),
			.w_dma = (uint32_t)(b_dma + ((size_t)bb * J->b_bat + (size_t)jc * b_tile) * ib),
			.out_dma = (uint32_t)(ctx->c.dma + ((size_t)bb * J->c_bat + (size_t)jc * c_tile + (size_t)i * Mt * Nc) * ob),
			.next_dma = last ? 0 : (uint32_t)(ctx->regcmd.dma + (size_t)(t + 1) * REGCMD_SLOTS * 8),
			.relu = J->relu_dpu,
		};
		emit_task(rc + (size_t)t * REGCMD_SLOTS, &d);
		struct rknpu_task td = {
			.flags = 0, .op_idx = 0, .enable_mask = 0xd, .int_mask = 0x300, .int_clear = 0x1ffff,
			.int_status = 0, .regcfg_amount = REGCFG_AMOUNT, .regcfg_offset = 0,
			.regcmd_addr = ctx->regcmd.dma + (size_t)t * REGCMD_SLOTS * 8,
		};
		tk[t] = td;
	}
	buf_sync(ctx, &ctx->regcmd, (size_t)TT * REGCMD_SLOTS * 8, RKNPU_MEM_SYNC_TO_DEVICE);
	buf_sync(ctx, &ctx->tasks, (size_t)TT * sizeof(struct rknpu_task), RKNPU_MEM_SYNC_TO_DEVICE);

	npu_lock(ctx);
	for (int s = 0; s < TT; s += per_submit) {
		int cnt = TT - s < per_submit ? TT - s : per_submit;
		if (submit(ctx, s, cnt, (cores == 3 && cnt >= 3) ? 3 : 1) < 0) { npu_unlock(ctx); return -1; }
	}
	npu_unlock(ctx);

	buf_sync(ctx, &ctx->c, c_bytes, RKNPU_MEM_SYNC_FROM_DEVICE);
	if (J->C) {
		t0 = now_us();
		pfor_batched(ctx, J, M, 64, unpack_range);
		ctx->stats.unpack_us += now_us() - t0;
	}
	return 0;
}

/* ------------------------------------------------------------------------------------
 * Precision front-ends
 * ---------------------------------------------------------------------------------- */
static float *scratch(rk_ctx *ctx, int which, size_t n) {
	if (ctx->scr_n[which] < n) {
		free(ctx->scr[which]);
		ctx->scr[which] = malloc(n * sizeof(float));
		ctx->scr_n[which] = ctx->scr[which] ? n : 0;
	}
	return ctx->scr[which];
}

/* per-row (or per-column) max-abs for every batch element into amax_part[batch][rows] */
static float amax_all(rk_ctx *ctx, job_t *J, int rows, range_fn fn) {
	int nb = J->batch > 1 ? J->batch : 1;
	J->amax_part = scratch(ctx, 0, (size_t)rows * nb);
	J->part_stride = rows;
	/* items are 64-row blocks; the scan functions index amax_part by row */
	pfor_batched(ctx, J, ceil_div(rows, 64), 1, fn);
	float mx = 0.f;
	for (size_t i = 0; i < (size_t)rows * nb; i++) if (J->amax_part[i] > mx) mx = J->amax_part[i];
	return mx;
}

/* split x*s into hi = fp16(x*s) and lo = fp16((x*s - hi) * 2^11), stored as fp32 */
static void split_a_range(void *v, int b, int e) {
	job_t *J = v;   /* reuses fields: A source, C = destination [M][2K] */
	for (int m = b; m < e; m++) {
		float *d = J->C + (size_t)m * 2 * J->K;
		for (int k = 0; k < J->K; k++) {
			float x = J->A[(int64_t)m * J->sa_m + (int64_t)k * J->sa_k] * J->a_scale;
			float hi = (float)(f16)x;
			d[k] = hi;
			d[J->K + k] = (float)(f16)((x - hi) * 2048.f);
		}
	}
}
static void split_b_range(void *v, int b, int e) {
	job_t *J = v;   /* destination [2K][N]: rows 0..K-1 = lo, rows K..2K-1 = hi */
	for (int k = b; k < e; k++) {
		float *lo = J->C + (size_t)k * J->N, *hi = J->C + (size_t)(J->K + k) * J->N;
		for (int n = 0; n < J->N; n++) {
			float x = J->B[(int64_t)k * J->sb_k + (int64_t)n * J->sb_n] * J->b_scale;
			float h = (float)(f16)x;
			hi[n] = h;
			lo[n] = (float)(f16)((x - h) * 2048.f);
		}
	}
}

static int gemm_front(rk_ctx *ctx, int prec, int batch, int M, int N, int K,
                      const float *A, int64_t sa_b, int64_t sa_m, int64_t sa_k,
                      const float *B, int64_t sb_b, int64_t sb_k, int64_t sb_n,
                      float *C, int64_t sc_b, int64_t ldc, int accumulate) {
	job_t J = {
		.A = A, .B = B, .sa_m = sa_m, .sa_k = sa_k, .sb_k = sb_k, .sb_n = sb_n,
		.M = M, .N = N, .K = K, .C = C, .ldc = ldc, .accumulate = accumulate,
		.a_scale = 1.f, .b_scale = 1.f, .out_mul = 1.f,
		.batch = batch, .sa_b = sa_b, .sb_b = sb_b, .sc_b = sc_b,
		.bias = ctx->epi_bias, .relu = ctx->epi_relu,
	};
	int r = -1;
	double t0;
	/* packed-weight cache: B is the cached operand (single GEMM, single-pass precisions) */
	const int wd[2] = { K, N };
	struct wc_entry *hit = NULL, *fresh = NULL;
	if (ctx->wc_key && batch == 1 && prec != RK_PREC_FP16X3) {
		hit = wc_find(ctx, 1, prec, wd, 2);
		if (hit && !hit->packed) { rk_wcache_drop(ctx, hit->key); hit = NULL; }
	}
	switch (prec) {
	case RK_PREC_FP16:
	case RK_PREC_BF16:
	case RK_PREC_FP16_OUT16:
		J.dt = prec == RK_PREC_BF16 ? &DT_BF16_F32 : prec == RK_PREC_FP16 ? &DT_F16_F32 : &DT_F16_F16;
		J.cv = prec == RK_PREC_BF16 ? CV_BF16 : CV_F16;
		if (prec == RK_PREC_FP16 && ctx->autoscale) {
			/* exact power-of-two scaling keeps both operands inside fp16 range */
			t0 = now_us();
			J.a_scale = pow2_scale(amax_all(ctx, &J, M, amax_a_rows));
			J.b_scale = hit ? hit->scale : pow2_scale(amax_all(ctx, &J, N, amax_b_cols));
			J.out_mul = 1.f / (J.a_scale * J.b_scale);
			J.amax_part = NULL;
			ctx->stats.pack_us += now_us() - t0;
		}
		if (!hit && ctx->wc_key && batch == 1) {
			fresh = wc_new(ctx, 1, prec, wd, 2, (size_t)round_up(K, 32) * round_up(N, 32) * 2);
			if (fresh) fresh->scale = J.b_scale;
		}
		J.wc = hit ? hit : fresh;
		r = gemm_engine(ctx, &J);
		break;
	case RK_PREC_INT8: {
		/* symmetric per-row (A) / per-column (B) quantization, exact int32 accumulation */
		J.dt = &DT_I8_I32;
		J.cv = CV_I8;
		t0 = now_us();
		size_t RA = (size_t)M * batch, CB = (size_t)N * batch;
		float *ra = scratch(ctx, 1, RA), *cb = scratch(ctx, 2, 2 * CB);
		if (!ra || !cb) { set_err(ctx, "out of memory"); return -1; }
		amax_all(ctx, &J, M, amax_a_rows);
		for (size_t m = 0; m < RA; m++) ra[m] = J.amax_part[m] > 0.f ? 127.f / J.amax_part[m] : 1.f;
		if (hit) {
			memcpy(cb, hit->col, 2 * CB * sizeof(float));
		} else {
			amax_all(ctx, &J, N, amax_b_cols);
			for (size_t n = 0; n < CB; n++) {
				cb[n] = J.amax_part[n] > 0.f ? 127.f / J.amax_part[n] : 1.f;
				cb[CB + n] = 1.f / cb[n];
			}
			if (ctx->wc_key && batch == 1) {
				fresh = wc_new(ctx, 1, prec, wd, 2, (size_t)round_up(K, 32) * round_up(N, 32));
				if (fresh && (fresh->col = malloc(2 * CB * sizeof(float))))
					memcpy(fresh->col, cb, 2 * CB * sizeof(float));
				else if (fresh) { rk_wcache_drop(ctx, fresh->key); fresh = NULL; }
			}
		}
		J.wc = hit ? hit : fresh;
		J.a_row = ra;
		J.b_col = cb;
		J.b_inv = cb + CB;
		J.amax_part = NULL;
		ctx->stats.pack_us += now_us() - t0;
		r = gemm_engine(ctx, &J);
		break;
	}
	case RK_PREC_FP16X3: {
		if (batch > 1) {   /* one batch element at a time */
			r = 0;
			for (int bb = 0; bb < batch && !r; bb++)
				r = gemm_front(ctx, prec, 1, M, N, K, A + bb * sa_b, 0, sa_m, sa_k, B + bb * sb_b, 0, sb_k, sb_n,
				               C + bb * sc_b, 0, ldc, accumulate);
			break;
		}
		J.batch = 1;
		/* C = Ah*Bh + 2^-11 * (Ah*Bl' + Al'*Bh): two NPU GEMMs, ~fp32 accuracy */
		t0 = now_us();
		float sa = pow2_scale(amax_all(ctx, &J, M, amax_a_rows));
		float sb = pow2_scale(amax_all(ctx, &J, N, amax_b_cols));
		float *A2 = scratch(ctx, 1, (size_t)M * 2 * K), *B2 = scratch(ctx, 2, (size_t)2 * K * N);
		if (!A2 || !B2) { set_err(ctx, "out of memory"); return -1; }
		job_t S = J;
		S.a_scale = sa; S.b_scale = sb;
		S.C = A2; parallel_for(ctx, M, 8, split_a_range, &S);
		S.C = B2; parallel_for(ctx, K, 8, split_b_range, &S);
		ctx->stats.pack_us += now_us() - t0;
		J.amax_part = NULL;
		/* pass 1: Ah (first K columns of A2) x Bh (rows K..2K-1 of B2) */
		job_t P = J;
		P.bias = NULL; P.relu = 0;
		P.dt = &DT_F16_F32; P.cv = CV_F16;
		P.A = A2; P.sa_m = 2 * K; P.sa_k = 1;
		P.B = B2 + (size_t)K * N; P.sb_k = N; P.sb_n = 1;
		P.out_mul = 1.f / (sa * sb);
		r = gemm_engine(ctx, &P);
		if (r) break;
		/* pass 2: [Ah | Al'] x [Bl' ; Bh] over 2K, scaled by 2^-11 and added */
		job_t Q = J;
		Q.dt = &DT_F16_F32; Q.cv = CV_F16;
		Q.A = A2; Q.sa_m = 2 * K; Q.sa_k = 1; Q.K = 2 * K;
		Q.B = B2; Q.sb_k = N; Q.sb_n = 1;
		Q.out_mul = 1.f / (2048.f * sa * sb);
		Q.accumulate = 1;
		r = gemm_engine(ctx, &Q);
		break;
	}
	default:
		set_err(ctx, "unknown precision %d", prec);
		return -1;
	}
	return r;
}

/* Split calls whose NPU buffers would exceed max_buf: batch halves first, then row bands. */
static int gemm_split(rk_ctx *ctx, int prec, int batch, int M, int N, int K,
                      const float *A, int64_t sa_b, int64_t sa_m, int64_t sa_k,
                      const float *B, int64_t sb_b, int64_t sb_k, int64_t sb_n,
                      float *C, int64_t sc_b, int64_t ldc, int accumulate) {
	size_t Kp = (size_t)round_up(K, 32), Np = (size_t)round_up(N, 32), Mp = (size_t)round_up(M, 4);
	size_t need = (size_t)batch * (Mp * Kp * 2 + Np * Kp * 2 + Mp * Np * 4);
	if (prec == RK_PREC_FP16X3) need *= 2;
	if (need > ctx->max_buf && batch > 1) {
		int b1 = batch / 2;
		int r = gemm_split(ctx, prec, b1, M, N, K, A, sa_b, sa_m, sa_k, B, sb_b, sb_k, sb_n, C, sc_b, ldc, accumulate);
		if (r) return r;
		return gemm_split(ctx, prec, batch - b1, M, N, K, A + b1 * sa_b, sa_b, sa_m, sa_k, B + b1 * sb_b, sb_b,
		                  sb_k, sb_n, C + b1 * sc_b, sc_b, ldc, accumulate);
	}
	if (need > ctx->max_buf && M > 64) {
		int m1 = (M / 2 + 3) & ~3;
		int r = gemm_split(ctx, prec, batch, m1, N, K, A, sa_b, sa_m, sa_k, B, sb_b, sb_k, sb_n, C, sc_b, ldc, accumulate);
		if (r) return r;
		return gemm_split(ctx, prec, batch, M - m1, N, K, A + (int64_t)m1 * sa_m, sa_b, sa_m, sa_k, B, sb_b, sb_k, sb_n,
		                  C + (int64_t)m1 * ldc, sc_b, ldc, accumulate);
	}
	return gemm_front(ctx, prec, batch, M, N, K, A, sa_b, sa_m, sa_k, B, sb_b, sb_k, sb_n, C, sc_b, ldc, accumulate);
}

int rk_gemm_batched(rk_ctx *ctx, int prec, int batch, int M, int N, int K,
                    const float *A, int64_t sa_b, int64_t sa_m, int64_t sa_k,
                    const float *B, int64_t sb_b, int64_t sb_k, int64_t sb_n,
                    float *C, int64_t sc_b, int64_t ldc, int accumulate) {
	double t_start = now_us();
	memset(&ctx->stats, 0, sizeof(ctx->stats));
	if (M <= 0 || N <= 0 || K <= 0 || batch <= 0) { set_err(ctx, "bad shape"); return -1; }
	int r = gemm_split(ctx, prec, batch, M, N, K, A, sa_b, sa_m, sa_k, B, sb_b, sb_k, sb_n, C, sc_b, ldc, accumulate);
	ctx->stats.total_us = now_us() - t_start;
	return r;
}

int rk_gemm_ex(rk_ctx *ctx, int prec, int M, int N, int K,
               const float *A, int64_t sa_m, int64_t sa_k,
               const float *B, int64_t sb_k, int64_t sb_n,
               float *C, int64_t ldc, int accumulate) {
	return rk_gemm_batched(ctx, prec, 1, M, N, K, A, 0, sa_m, sa_k, B, 0, sb_k, sb_n, C, 0, ldc, accumulate);
}


int rk_gemm_f32(rk_ctx *ctx, int M, int N, int K,
                const float *A, int64_t sa_m, int64_t sa_k,
                const float *B, int64_t sb_k, int64_t sb_n,
                float *C, int64_t ldc, int accumulate) {
	return rk_gemm_ex(ctx, RK_PREC_FP16, M, N, K, A, sa_m, sa_k, B, sb_k, sb_n, C, ldc, accumulate);
}

/* ------------------------------------------------------------------------------------
 * Raw probe: one task with caller-chosen precision codes and pre-laid-out bytes.
 * ---------------------------------------------------------------------------------- */
int rk_probe_raw(rk_ctx *ctx, const rk_probe_cfg *cfg, int M, int N, int K,
                 const void *a_rows, const void *w_bytes, void *c_rows) {
	dtype_t dt = { cfg->in_prec, cfg->proc_prec, cfg->out_prec, cfg->in_bytes, cfg->out_bytes,
	               cfg->kgroup, cfg->qd_en, cfg->bs_ow, cfg->cvt_scale, cfg->cvt_shift, cfg->surf_add };
	size_t a_bytes = (size_t)M * K * dt.in_bytes, w_bytes_n = (size_t)N * K * dt.in_bytes;
	size_t c_bytes = (size_t)M * N * dt.out_bytes;
	uint32_t dflags = MEM_FLAG_400 | RKNPU_MEM_CACHEABLE | RKNPU_MEM_NON_CONTIGUOUS;
	if (buf_ensure(ctx, &ctx->a, a_bytes, dflags) || buf_ensure(ctx, &ctx->b, w_bytes_n, dflags) ||
	    buf_ensure(ctx, &ctx->c, c_bytes, dflags) || buf_ensure(ctx, &ctx->regcmd, REGCMD_SLOTS * 8, dflags) ||
	    buf_ensure(ctx, &ctx->tasks, sizeof(struct rknpu_task), dflags | RKNPU_MEM_KERNEL_MAPPING))
		return -1;
	memcpy(ctx->a.ptr, a_rows, a_bytes);
	memcpy(ctx->b.ptr, w_bytes, w_bytes_n);
	memset(ctx->c.ptr, 0x7f, c_bytes);   /* poison, so "not written" is visible */
	buf_sync(ctx, &ctx->a, a_bytes, RKNPU_MEM_SYNC_TO_DEVICE);
	buf_sync(ctx, &ctx->b, w_bytes_n, RKNPU_MEM_SYNC_TO_DEVICE);
	buf_sync(ctx, &ctx->c, c_bytes, RKNPU_MEM_SYNC_TO_DEVICE);
	int dbanks = ceil_div((M + 1) * K * dt.in_bytes, CBUF_BANK_SIZE);
	task_desc d = { .dt = &dt, .m = M, .k = K, .n = N, .grains = M + 1, .dbanks = dbanks,
	                .wbanks = CBUF_BANKS - dbanks, .reuse = 0, .in_dma = (uint32_t)ctx->a.dma,
	                .w_dma = (uint32_t)ctx->b.dma, .out_dma = (uint32_t)ctx->c.dma, .next_dma = 0 };
	emit_task(ctx->regcmd.ptr, &d);
	struct rknpu_task td = { .enable_mask = 0xd, .int_mask = 0x300, .int_clear = 0x1ffff,
	                         .regcfg_amount = REGCFG_AMOUNT, .regcmd_addr = ctx->regcmd.dma };
	*(struct rknpu_task *)ctx->tasks.ptr = td;
	buf_sync(ctx, &ctx->regcmd, REGCMD_SLOTS * 8, RKNPU_MEM_SYNC_TO_DEVICE);
	buf_sync(ctx, &ctx->tasks, sizeof(td), RKNPU_MEM_SYNC_TO_DEVICE);
	if (submit(ctx, 0, 1, 1) < 0) return -1;
	buf_sync(ctx, &ctx->c, c_bytes, RKNPU_MEM_SYNC_FROM_DEVICE);
	memcpy(c_rows, ctx->c.ptr, c_bytes);
	return 0;
}

/* ------------------------------------------------------------------------------------
 * Raw convolution probe: one task, NHWC fp16 input (whole image), caller-laid-out
 * weights, NHWC fp32 output. Used to work out the convolution register programming.
 * ---------------------------------------------------------------------------------- */
int rk_conv_raw(rk_ctx *ctx, const rk_conv_cfg *c, const void *in_nhwc, const void *w_bytes, void *out_nhwc) {
	const dtype_t *dt = &DT_F16_F32;
	/* c->c must be a multiple of 8 (fp16 atoms), c->n a multiple of 16 */
	size_t in_b = (size_t)c->h * c->w * c->c * 2, w_b = (size_t)(c->dw ? 1 : c->n) * c->kh * c->kw * c->c * 2;
	size_t out_b = (size_t)c->oh * c->ow * c->n * 4;
	uint32_t dflags = MEM_FLAG_400 | RKNPU_MEM_CACHEABLE | RKNPU_MEM_NON_CONTIGUOUS;
	if (buf_ensure(ctx, &ctx->a, in_b, dflags) || buf_ensure(ctx, &ctx->b, w_b, dflags) ||
	    buf_ensure(ctx, &ctx->c, out_b, dflags) || buf_ensure(ctx, &ctx->regcmd, REGCMD_SLOTS * 8, dflags) ||
	    buf_ensure(ctx, &ctx->tasks, sizeof(struct rknpu_task), dflags | RKNPU_MEM_KERNEL_MAPPING))
		return -1;
	memcpy(ctx->a.ptr, in_nhwc, in_b);
	memcpy(ctx->b.ptr, w_bytes, w_b);
	memset(ctx->c.ptr, 0x7f, out_b);
	buf_sync(ctx, &ctx->a, in_b, RKNPU_MEM_SYNC_TO_DEVICE);
	buf_sync(ctx, &ctx->b, w_b, RKNPU_MEM_SYNC_TO_DEVICE);
	buf_sync(ctx, &ctx->c, out_b, RKNPU_MEM_SYNC_TO_DEVICE);
	int dbanks = c->dbanks > 0 ? c->dbanks : ceil_div((int)in_b + c->w * c->c * 2, CBUF_BANK_SIZE);
	int wbanks = CBUF_BANKS - dbanks;
	task_desc d = { .dt = dt, .m = c->h, .k = c->c, .n = c->n, .grains = c->grains > 0 ? c->grains : c->h + 1,
	                .dbanks = dbanks, .wbanks = wbanks, .reuse = 0, .in_dma = (uint32_t)ctx->a.dma,
	                .w_dma = (uint32_t)ctx->b.dma, .out_dma = (uint32_t)ctx->c.dma, .next_dma = 0,
	                .in_w = c->w, .in_h = c->h, .out_w = c->ow, .out_h = c->oh, .kw = c->kw, .kh = c->kh,
	                .sx = c->sx, .sy = c->sy, .pl = c->pl, .pt = c->pt, .nc1hwc2 = c->layout,
	                .dw = c->dw, .relu = c->relu };
	emit_task(ctx->regcmd.ptr, &d);
	struct rknpu_task td = { .enable_mask = 0xd, .int_mask = 0x300, .int_clear = 0x1ffff,
	                         .regcfg_amount = REGCFG_AMOUNT, .regcmd_addr = ctx->regcmd.dma };
	*(struct rknpu_task *)ctx->tasks.ptr = td;
	buf_sync(ctx, &ctx->regcmd, REGCMD_SLOTS * 8, RKNPU_MEM_SYNC_TO_DEVICE);
	buf_sync(ctx, &ctx->tasks, sizeof(td), RKNPU_MEM_SYNC_TO_DEVICE);
	npu_lock(ctx);
	int r = submit(ctx, 0, 1, 1);
	npu_unlock(ctx);
	if (r < 0) return -1;
	buf_sync(ctx, &ctx->c, out_b, RKNPU_MEM_SYNC_FROM_DEVICE);
	memcpy(out_nhwc, ctx->c.ptr, out_b);
	return 0;
}

/* ------------------------------------------------------------------------------------
 * Native convolution: fp32 NCHW in/out, NC1HWC2 on the NPU
 * ---------------------------------------------------------------------------------- */
typedef struct {
	const float *x, *w;
	float *y;
	int N, C, H, W, O, kh, kw, oh, ow;
	int G, Cg, Og;                 /* groups, input / output channels per group */
	int Cp, Cc, nk, Op;
	f16 *xp, *wp;                  /* packed fp16 (or bf16 bits) */
	float *yp;
	int bf16;
	float xs, ws, out_mul;
	int accumulate;
	const float *bias;             /* epilogue: + bias[o] */
	int relu, relu_dpu;
	size_t x_img, x_chunk, w_chunk, y_img, y_chunk;   /* elements (per group) */
	size_t x_grp, w_grp, y_grp;                       /* elements per group */
} conv_job;

static inline uint16_t h16(conv_job *J, float v) {
	if (J->bf16) return f32_to_bf16(v);
	f16 h = (f16)v;
	uint16_t u;
	memcpy(&u, &h, 2);
	return u;
}

/* x: items (group, b, chunk j, c1 group of 8 channels) -> [C1][H][W][8] */
static void conv_pack_x(void *v, int b, int e) {
	conv_job *J = v;
	const int HW = J->H * J->W, g8 = J->Cc / 8;
	for (int it = b; it < e; it++) {
		int c1 = it % g8, j = (it / g8) % J->nk, bb = (it / (g8 * J->nk)) % J->N, gr = it / (g8 * J->nk * J->N);
		uint16_t *d = (uint16_t *)J->xp + (size_t)gr * J->x_grp + (size_t)bb * J->x_img + (size_t)j * J->x_chunk +
		              (size_t)c1 * HW * 8;
		int c0 = j * J->Cc + c1 * 8;
		const float *src[8];
		for (int t = 0; t < 8; t++)
			src[t] = (c0 + t < J->Cg) ? J->x + ((size_t)bb * J->C + (size_t)gr * J->Cg + c0 + t) * HW : NULL;
		for (int p = 0; p < HW; p++)
			for (int t = 0; t < 8; t++)
				d[(size_t)p * 8 + t] = h16(J, src[t] ? src[t][p] * J->xs : 0.f);
	}
}

/* w (O, Cg, kh, kw): items (group, chunk j, 16-kernel group) -> [Cc/32][kh][kw][16][32] */
static void conv_pack_w(void *v, int b, int e) {
	conv_job *J = v;
	const int ng = J->Op / 16, khw = J->kh * J->kw;
	for (int it = b; it < e; it++) {
		int og = it % ng, j = (it / ng) % J->nk, gr = it / (ng * J->nk);
		uint16_t *d = (uint16_t *)J->wp + (size_t)gr * J->w_grp + (size_t)j * J->w_chunk + (size_t)og * 16 * J->Cc * khw;
		size_t n = 0;
		for (int cb = 0; cb < J->Cc / 32; cb++)
			for (int ky = 0; ky < J->kh; ky++)
				for (int kx = 0; kx < J->kw; kx++)
					for (int o2 = 0; o2 < 16; o2++)
						for (int c2 = 0; c2 < 32; c2++) {
							int o = og * 16 + o2, c = j * J->Cc + cb * 32 + c2;
							float v0 = (o < J->Og && c < J->Cg)
							           ? J->w[((((size_t)gr * J->Og + o) * J->Cg + c) * J->kh + ky) * J->kw + kx] * J->ws : 0.f;
							d[n++] = h16(J, v0);
						}
	}
}

/* y: items (b, o) -> NCHW plane, summing channel chunks; source [Op/4][oh][ow][4] per group */
static void conv_unpack_y(void *v, int b, int e) {
	conv_job *J = v;
	const int OHW = J->oh * J->ow;
	const int relu = J->relu && !J->relu_dpu;
	for (int it = b; it < e; it++) {
		int o = it % J->O, bb = it / J->O, gr = o / J->Og, ol = o % J->Og;
		float *d = J->y + ((size_t)bb * J->O + o) * OHW;
		const float *src = J->yp + (size_t)gr * J->y_grp + (size_t)bb * J->y_img + (size_t)(ol / 4) * OHW * 4 + (ol & 3);
		const float bias = J->bias ? J->bias[o] : 0.f;
		for (int p = 0; p < OHW; p++) {
			float acc = 0.f;
			for (int j = 0; j < J->nk; j++) acc += src[(size_t)j * J->y_chunk + (size_t)p * 4];
			acc *= J->out_mul;
			if (J->accumulate) acc += d[p];
			acc += bias;
			if (relu && acc < 0.f) acc = 0.f;
			d[p] = acc;
		}
	}
}

typedef struct { const float *p; size_t n; uint32_t *part; } amax_job;
static void amax_flat(void *v, int b, int e) {
	amax_job *A = v;
	for (int it = b; it < e; it++) {
		size_t i0 = (size_t)it * 65536, i1 = i0 + 65536 < A->n ? i0 + 65536 : A->n;
		const uint32_t *s = (const uint32_t *)A->p;
		uint32_t x = 0;
		for (size_t i = i0; i < i1; i++) { uint32_t a = s[i] & 0x7fffffffu; x = a > x ? a : x; }
		A->part[it] = x;
	}
}
static float amax_par(rk_ctx *ctx, const float *p, size_t n) {
	int items = (int)((n + 65535) / 65536);
	uint32_t *part = (uint32_t *)scratch(ctx, 0, (size_t)items);
	if (!part) { float m = 0.f; for (size_t i = 0; i < n; i++) m = fabsf(p[i]) > m ? fabsf(p[i]) : m; return m; }
	amax_job A = { p, n, part };
	parallel_for(ctx, items, 1, amax_flat, &A);
	uint32_t m = 0;
	for (int i = 0; i < items; i++) m = part[i] > m ? part[i] : m;
	return bitsf(m);
}

int rk_conv2d_ex(rk_ctx *ctx, int prec, int N, int C, int H, int W, const float *x,
                 int O, int kh, int kw, const float *w, int sy, int sx, int pt, int pl,
                 int oh, int ow, float *y, int accumulate, int groups) {
	if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || O <= 0 || kh <= 0 || kw <= 0 || oh <= 0 || ow <= 0 ||
	    kh > 31 || kw > 31 || sy < 1 || sx < 1 || sy > 7 || sx > 7 || pt < 0 || pl < 0 || pt > 15 || pl > 15 ||
	    W > 2047 || H > 2047 || groups < 1 || C % groups || O % groups) {
		set_err(ctx, "unsupported convolution geometry");
		return -1;
	}
	const dtype_t *dt = prec == RK_PREC_BF16 ? &DT_BF16_F32 : &DT_F16_F32;
	const int G = groups, Cg = C / G, Og = O / G;
	conv_job J = { .x = x, .w = w, .y = y, .N = N, .C = C, .H = H, .W = W, .O = O, .kh = kh, .kw = kw,
	               .oh = oh, .ow = ow, .G = G, .Cg = Cg, .Og = Og, .bf16 = prec == RK_PREC_BF16,
	               .xs = 1.f, .ws = 1.f, .out_mul = 1.f, .accumulate = accumulate,
	               .bias = ctx->epi_bias, .relu = ctx->epi_relu };
	const int khw = kh * kw;
	/* channel chunks: a 16-kernel weight group must fit in <= 8 banks, and one input row in the rest */
	int Cp = round_up(Cg, 32), Cc = Cp;
	while (Cc > 32 && (16L * khw * Cc * 2 > 8L * CBUF_BANK_SIZE)) Cc = round_up(Cc / 2, 32);
	int nk = ceil_div(Cp, Cc);
	Cc = round_up(ceil_div(Cp, nk), 32);
	Cp = Cc * nk;
	int Op = round_up(Og, 16);
	if (Op > 4096) { set_err(ctx, "too many output channels (%d) for one convolution", O); return -1; }
	J.Cp = Cp; J.Cc = Cc; J.nk = nk; J.Op = Op;
	/* CBUF split */
	long wgroup = 16L * khw * Cc * 2, wall = (long)Op * khw * Cc * 2;
	int wbanks = (int)ceil_div((int)(wall < 4L * CBUF_BANK_SIZE ? wall : (wgroup > 4L * CBUF_BANK_SIZE ? wgroup : 4L * CBUF_BANK_SIZE)), CBUF_BANK_SIZE);
	if (wbanks < 1) wbanks = 1;
	int dbanks_max = CBUF_BANKS - wbanks;
	long entries_row = (W * 2L * (Cc / 8) + 7) / 8;                  /* CBUF entries (128 B) per input row */
	long rows_fit = (dbanks_max * 256L) / entries_row;                /* input rows that fit */
	if (rows_fit < kh + 2) { set_err(ctx, "image row too wide for CBUF (W=%d, C=%d)", W, Cg); return -1; }
	/* output rows per slice: (rows-1)*sy + kh input rows (+1 preload) must fit */
	int orows = (int)((rows_fit - 1 - kh) / sy) + 1;
	/* FEATURE_GRAINS (input rows + 1) is a 10-bit field */
	int orows_grains = (1022 - kh) / sy + 1;
	if (orows > orows_grains) orows = orows_grains;
	if (orows > oh) orows = oh;
	int nslice = ceil_div(oh, orows);
	orows = ceil_div(oh, nslice);
	int max_in_rows = (orows - 1) * sy + kh;
	if (max_in_rows > H) max_in_rows = H;
	int dbanks_all = ceil_div((int)(entries_row * (max_in_rows + 1)), 256);
	if (dbanks_all > dbanks_max) dbanks_all = dbanks_max;
	wbanks = CBUF_BANKS - dbanks_all;
	int weights_fit = wall <= (long)wbanks * CBUF_BANK_SIZE;
	int T = G * N * nk * nslice;
	if (T > 64 * PC_TASK_LIMIT) { set_err(ctx, "convolution needs too many NPU tasks (%d)", T); return -1; }

	J.x_chunk = (size_t)Cc * H * W; J.x_img = J.x_chunk * nk; J.x_grp = J.x_img * N;
	J.w_chunk = (size_t)Op * khw * Cc; J.w_grp = J.w_chunk * nk;
	J.y_chunk = (size_t)Op * oh * ow; J.y_img = J.y_chunk * nk; J.y_grp = J.y_img * N;
	size_t xb = J.x_grp * G * 2, wb = J.w_grp * G * 2, yb = J.y_grp * G * 4;
	uint32_t dflags = MEM_FLAG_400 | RKNPU_MEM_CACHEABLE | RKNPU_MEM_NON_CONTIGUOUS;
	if (xb + wb + yb > ctx->max_buf * 2 && N > 1) {
		/* split the batch to bound NPU memory (the weight cache entry is reused by both halves) */
		int n1 = N / 2;
		int r = rk_conv2d_ex(ctx, prec, n1, C, H, W, x, O, kh, kw, w, sy, sx, pt, pl, oh, ow, y, accumulate, groups);
		if (r) return r;
		return rk_conv2d_ex(ctx, prec, N - n1, C, H, W, x + (size_t)n1 * C * H * W, O, kh, kw, w, sy, sx, pt, pl, oh, ow,
		                    y + (size_t)n1 * O * oh * ow, accumulate, groups);
	}
	ctx->stats.tasks += T;

	/* packed-weight cache */
	const int wd[6] = { O, Cg, kh, kw, G, Cc };
	struct wc_entry *wc = NULL;
	int pack_w = 1;
	if (ctx->wc_key) {
		wc = wc_find(ctx, 2, prec, wd, 6);
		if (wc && wc->packed) pack_w = 0;
		else {
			if (wc) rk_wcache_drop(ctx, wc->key);
			wc = wc_new(ctx, 2, prec, wd, 6, wb);
			if (wc) wc = wc_alloc(ctx, wc, wb);
		}
	}
	if (buf_ensure(ctx, &ctx->a, xb, dflags) || (!wc && buf_ensure(ctx, &ctx->b, wb, dflags)) ||
	    buf_ensure(ctx, &ctx->c, yb, dflags) ||
	    buf_ensure(ctx, &ctx->regcmd, (size_t)T * REGCMD_SLOTS * 8, dflags) ||
	    buf_ensure(ctx, &ctx->tasks, (size_t)T * sizeof(struct rknpu_task), dflags | RKNPU_MEM_KERNEL_MAPPING))
		return -1;
	J.xp = ctx->a.ptr; J.wp = wc ? wc->buf.ptr : ctx->b.ptr; J.yp = ctx->c.ptr;
	const uint64_t w_dma = wc ? wc->buf.dma : ctx->b.dma;
	J.relu_dpu = J.relu && !J.bias && nk == 1 && !accumulate;

	double t0 = now_us();
	if (!J.bf16 && ctx->autoscale) {
		J.xs = pow2_scale(amax_par(ctx, x, (size_t)N * C * H * W));
		J.ws = pack_w ? pow2_scale(amax_par(ctx, w, (size_t)O * Cg * khw)) : wc->scale;
		J.out_mul = 1.f / (J.xs * J.ws);
	}
	parallel_for(ctx, G * N * nk * (Cc / 8), 1, conv_pack_x, &J);
	if (pack_w) parallel_for(ctx, G * nk * (Op / 16), 1, conv_pack_w, &J);
	ctx->stats.pack_us += now_us() - t0;
	buf_sync(ctx, &ctx->a, xb, RKNPU_MEM_SYNC_TO_DEVICE);
	if (pack_w) {
		buf_sync(ctx, wc ? &wc->buf : &ctx->b, wb, RKNPU_MEM_SYNC_TO_DEVICE);
		if (wc) { wc->packed = 1; wc->scale = J.ws; }
	} else {
		ctx->stats.cache_hits++;
	}

	uint64_t *rc = ctx->regcmd.ptr;
	struct rknpu_task *tk = ctx->tasks.ptr;
	int cores = ctx->mode == 2 ? 3 : 1;
	int per_submit = ctx->mode == 0 ? 1 : PC_TASK_LIMIT;
	for (int t = 0; t < T; t++) {
		int sl = t % nslice, j = (t / nslice) % nk, bb = (t / (nslice * nk)) % N, gr = t / (nslice * nk * N);
		int r0 = sl * orows, r1 = r0 + orows < oh ? r0 + orows : oh;
		int in0 = r0 * sy - pt, in1 = (r1 - 1) * sy - pt + kh;
		int ptop = in0 < 0 ? -in0 : 0;
		if (in0 < 0) in0 = 0;
		if (in1 > H) in1 = H;
		int sub_start = (t / per_submit) * per_submit;
		int sub_count = T - sub_start < per_submit ? T - sub_start : per_submit;
		int first = 0, last = 0;
		if (ctx->mode == 0) first = last = 1;
		else if (cores == 1 || sub_count < 3) { first = t == sub_start; last = t == sub_start + sub_count - 1; }
		else {
			int base = sub_start;
			for (int cc = 0; cc < 3; cc++) {
				int n = sub_count / 3 + (cc < sub_count % 3);
				if (t == base) first = 1;
				if (t == base + n - 1) last = 1;
				base += n;
			}
		}
		int in_rows = in1 - in0;
		/* the same CBUF split for every task: the weight-reuse bit assumes weights stay in
		 * the same banks, which breaks if an edge slice (fewer rows) moved the boundary.
		 * A new group (bb == 0 && sl == 0) brings new weights. */
		int dbanks = dbanks_all;
		size_t xo = (size_t)gr * J.x_grp + (size_t)bb * J.x_img + (size_t)j * J.x_chunk + (size_t)in0 * W * 8;
		size_t yo = (size_t)gr * J.y_grp + (size_t)bb * J.y_img + (size_t)j * J.y_chunk + (size_t)r0 * ow * 4;
		task_desc d = {
			.dt = dt, .m = in_rows, .k = Cc, .n = Op, .grains = in_rows + 1,
			.dbanks = dbanks, .wbanks = CBUF_BANKS - dbanks,
			.reuse = ctx->reuse && weights_fit && !first && (sl > 0 || bb > 0) && nk == 1,
			.in_dma = (uint32_t)(ctx->a.dma + xo * 2),
			.w_dma = (uint32_t)(w_dma + ((size_t)gr * J.w_grp + (size_t)j * J.w_chunk) * 2),
			.out_dma = (uint32_t)(ctx->c.dma + yo * 4),
			.next_dma = last ? 0 : (uint32_t)(ctx->regcmd.dma + (size_t)(t + 1) * REGCMD_SLOTS * 8),
			.in_w = W, .in_h = in_rows, .out_w = ow, .out_h = r1 - r0, .kw = kw, .kh = kh, .sx = sx, .sy = sy,
			.pl = pl, .pt = ptop, .nc1hwc2 = 1, .surf_in_h = H, .surf_out_hw = oh * ow, .relu = J.relu_dpu,
		};
		emit_task(rc + (size_t)t * REGCMD_SLOTS, &d);
		struct rknpu_task td = {
			.enable_mask = 0xd, .int_mask = 0x300, .int_clear = 0x1ffff,
			.regcfg_amount = REGCFG_AMOUNT, .regcmd_addr = ctx->regcmd.dma + (size_t)t * REGCMD_SLOTS * 8,
		};
		tk[t] = td;
	}
	buf_sync(ctx, &ctx->regcmd, (size_t)T * REGCMD_SLOTS * 8, RKNPU_MEM_SYNC_TO_DEVICE);
	buf_sync(ctx, &ctx->tasks, (size_t)T * sizeof(struct rknpu_task), RKNPU_MEM_SYNC_TO_DEVICE);
	npu_lock(ctx);
	for (int s0 = 0; s0 < T; s0 += per_submit) {
		int cnt = T - s0 < per_submit ? T - s0 : per_submit;
		if (submit(ctx, s0, cnt, (cores == 3 && cnt >= 3) ? 3 : 1) < 0) { npu_unlock(ctx); return -1; }
	}
	npu_unlock(ctx);
	buf_sync(ctx, &ctx->c, yb, RKNPU_MEM_SYNC_FROM_DEVICE);
	t0 = now_us();
	parallel_for(ctx, N * O, 4, conv_unpack_y, &J);
	ctx->stats.unpack_us += now_us() - t0;
	return 0;
}

int rk_conv2d_grouped(rk_ctx *ctx, int prec, int N, int C, int H, int W, const float *x,
                      int O, int kh, int kw, const float *w, int sy, int sx, int pt, int pl,
                      int oh, int ow, float *y, int accumulate, int groups) {
	double t_start = now_us();
	memset(&ctx->stats, 0, sizeof(ctx->stats));
	int r = rk_conv2d_ex(ctx, prec, N, C, H, W, x, O, kh, kw, w, sy, sx, pt, pl, oh, ow, y, accumulate, groups);
	ctx->stats.total_us = now_us() - t_start;
	return r;
}

int rk_conv2d(rk_ctx *ctx, int prec, int N, int C, int H, int W, const float *x,
              int O, int kh, int kw, const float *w, int sy, int sx, int pt, int pl,
              int oh, int ow, float *y, int accumulate) {
	return rk_conv2d_grouped(ctx, prec, N, C, H, W, x, O, kh, kw, w, sy, sx, pt, pl, oh, ow, y, accumulate, 1);
}
