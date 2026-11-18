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
	float   *scr[3];
	size_t   scr_n[3];
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
	ctx->mode = env_int("RKNPU_MODE", 2);
	ctx->threads = env_int("RKNPU_THREADS", 3);
	ctx->max_m = env_int("RKNPU_MAX_M", 1020);
	ctx->max_k = env_int("RKNPU_MAX_K", 8192);
	ctx->max_n = env_int("RKNPU_MAX_N", 4096);
	ctx->timeout_ms = env_int("RKNPU_TIMEOUT_MS", 3000);
	ctx->reuse = env_int("RKNPU_WEIGHT_REUSE", 1);
	ctx->autoscale = env_int("RKNPU_AUTOSCALE", 1);

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
	pool_destroy(ctx->pool);
	for (int i = 0; i < 3; i++) free(ctx->scr[i]);
	close(ctx->fd);
	free(ctx);
}

void rk_set_mode(rk_ctx *ctx, int mode) { ctx->mode = mode; }
void rk_set_threads(rk_ctx *ctx, int t) { ctx->threads = t > 0 ? t : 1; }
void rk_set_max_m_tile(rk_ctx *ctx, int m) { ctx->max_m = m >= 4 ? m & ~3 : 4; }
void rk_get_stats(rk_ctx *ctx, rk_stats *out) { *out = ctx->stats; }

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
} task_desc;

/* Register stream for one row-major matmul task (precisions from t->dt). */
static void emit_task(uint64_t *o, const task_desc *t) {
	const dtype_t *dt = t->dt;
	const uint32_t m = t->m, k = t->k, n = t->n;
	const uint32_t kb = k * (uint32_t)dt->in_bytes;                   /* bytes per row / kernel */
	const uint32_t notch = n * (uint32_t)dt->out_bytes / 16 - 1;      /* 16-byte output atoms per row - 1 */
	const uint32_t cbuf = (t->reuse ? CBUF_WEIGHT_REUSE : 0) | ((uint32_t)t->wbanks << 4) | (uint32_t)t->dbanks;
	/* bit 29: row-major feature input */
	const uint32_t conv1 = (1u << 29) | ((uint32_t)dt->proc_prec << 7) | ((uint32_t)dt->in_prec << 4);
	int i = 0;
#define R(blk, reg, val) (o[i++] = npuop((blk), (uint32_t)(val), (reg)))
	R(OP_REG_CNA, CNA_CBUF_CON0, cbuf);
	R(OP_REG_CNA, CNA_DCOMP_REGNUM, 0);
	R(OP_REG_CNA, CNA_DCOMP_CTRL, 0);
	R(OP_REG_CNA, CNA_CONV_CON1, conv1);
	R(OP_REG_DPU, DPU_S_POINTER, 0xE);
	R(OP_REG_CNA, CNA_CONV_CON1, conv1);
	R(OP_REG_CNA, CNA_CONV_CON2, ((uint32_t)t->grains & 0x3FF) << 4);
	R(OP_REG_CNA, CNA_CONV_CON3, (1 << 3) | 1);
	R(OP_REG_CNA, CNA_DATA_SIZE0, (1u << 16) | (m & 0x7FF));
	R(OP_REG_CNA, CNA_DATA_SIZE1, ((k - 1) << 16) | k);
	R(OP_REG_CNA, CNA_DATA_SIZE2, 1);
	R(OP_REG_CNA, CNA_DATA_SIZE3, m & 0x3FFFF);
	R(OP_REG_CNA, CNA_WEIGHT_SIZE0, kb * n);
	R(OP_REG_CNA, CNA_WEIGHT_SIZE1, kb & 0x7FFFF);
	R(OP_REG_CNA, CNA_WEIGHT_SIZE2, (1u << 24) | (1u << 16) | (n & 0x3FFF));
	R(OP_REG_CNA, CNA_CBUF_CON0, cbuf);
	R(OP_REG_CNA, CNA_CBUF_CON1, ((kb + 63) / 64) & 0x1FFF);
	R(OP_REG_CNA, CNA_CVT_CON0, (1 << 3) | (1 << 1) | 1);
	R(OP_REG_CNA, CNA_CVT_CON1, 1u << 16);
	R(OP_REG_CNA, CNA_CVT_CON2, 1u << 16);
	R(OP_REG_CNA, CNA_CVT_CON3, 1u << 16);
	R(OP_REG_CNA, CNA_CVT_CON4, 1u << 16);
	R(OP_REG_CNA, CNA_FC_CON0, 0);
	R(OP_REG_CNA, CNA_FC_CON1, 0);
	R(OP_REG_CNA, CNA_PAD_CON0, 0);
	R(OP_REG_CNA, CNA_FEATURE_DATA_ADDR, t->in_dma);
	R(OP_REG_CNA, CNA_FC_CON2, 0);
	R(OP_REG_CNA, CNA_DMA_CON0, (15u << 16) | 15u);
	R(OP_REG_CNA, CNA_DMA_CON1, kb / 16);        /* line stride: one row of A, in 16-byte units */
	R(OP_REG_CNA, CNA_DMA_CON2, 0);              /* surface stride: channel groups are contiguous */
	R(OP_REG_CNA, CNA_FC_DATA_SIZE0, (1u << 16) | (m & 0x7FF));
	R(OP_REG_CNA, CNA_FC_DATA_SIZE1, k & 0xFFFF);
	R(OP_REG_CNA, CNA_DCOMP_CTRL, 0);
	R(OP_REG_CNA, CNA_DCOMP_REGNUM, 0);
	R(OP_REG_CNA, CNA_DCOMP_ADDR0, t->w_dma);
	for (int r = 0; r < 16; r++) R(OP_REG_CNA, CNA_DCOMP_AMOUNT0 + 4 * r, 0);
	R(OP_REG_CNA, CNA_CVT_CON5, 0);
	R(OP_REG_CNA, CNA_PAD_CON1, 0);

	R(OP_REG_CORE, CORE_MISC_CFG, ((uint32_t)dt->proc_prec << 8) | (uint32_t)dt->qd_en);
	R(OP_REG_CORE, CORE_DATAOUT_SIZE_0, (m - 1) << 16);
	R(OP_REG_CORE, CORE_DATAOUT_SIZE_1, n - 1);
	R(OP_REG_CORE, CORE_CLIP_TRUNCATE, 0);
	R(OP_REG_CORE, CORE_3030, 0);

	R(OP_REG_DPU, DPU_FEATURE_MODE_CFG, (15u << 5) | (2u << 1));
	R(OP_REG_DPU, DPU_DATA_FORMAT, ((uint32_t)dt->out_prec << 29) | ((uint32_t)dt->proc_prec << 26) | (uint32_t)dt->proc_prec);
	R(OP_REG_DPU, DPU_OFFSET_PEND, 0);
	R(OP_REG_DPU, DPU_DST_BASE_ADD, t->out_dma);
	R(OP_REG_DPU, DPU_DST_SURF_STRIDE, 1u << 4);  /* row-major output: 4-float atoms are adjacent */
	R(OP_REG_DPU, DPU_DATA_CUBE_WIDTH, 0);
	R(OP_REG_DPU, DPU_DATA_CUBE_HEIGHT, (m - 1) & 0x1FFF);
	R(OP_REG_DPU, DPU_DATA_CUBE_NOTCH_ADDR, (notch << 16) | notch);
	R(OP_REG_DPU, DPU_DATA_CUBE_CHANNEL, ((n - 1) << 16) | (n - 1));
	R(OP_REG_DPU, DPU_BS_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1);
	R(OP_REG_DPU, DPU_BS_ALU_CFG, 0);
	R(OP_REG_DPU, DPU_BS_MUL_CFG, 0);
	R(OP_REG_DPU, DPU_BS_RELUX_CMP_VALUE, 0);
	R(OP_REG_DPU, DPU_BS_OW_CFG, dt->bs_ow);
	R(OP_REG_DPU, DPU_BS_OW_OP, 0);
	R(OP_REG_DPU, DPU_WDMA_SIZE_0, (n - 1) & 0x1FFF);
	R(OP_REG_DPU, DPU_WDMA_SIZE_1, ((m - 1) & 0x1FFF) << 16);
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
	R(OP_REG_DPU, DPU_SURFACE_ADD, dt->surf_add << 4);
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
	float r = __builtin_roundf(x);
	if (r > 127.f) r = 127.f;
	if (r < -127.f) r = -127.f;
	return (int8_t)r;
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
	float *C;
	int64_t ldc;
	int accumulate;
	float out_mul;                        /* applied to every output (undo input scales) */
	float *amax_part;                     /* scratch for max-abs reductions */
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

/* fast path for the common fp16 case with B read along k (sb_k == 1, i.e. Linear weights) */
static void pack_b_f16_range(void *v, int b, int e) {
	job_t *J = v;
	const int Kc = J->Kc, Nc = J->Nc, K = J->K, N = J->N;
	const float sc = J->b_scale;
	for (int it = b; it < e; it++) {
		int jc = it / (Nc / 16), ng = it % (Nc / 16);
		int j = jc / J->nn, c = jc % J->nn;
		f16 *dst = (f16 *)J->bp + (size_t)jc * J->b_tile + (size_t)ng * 16 * Kc;
		if (J->sb_n == 1 && J->sb_k != 1) {
			for (int cb = 0; cb < Kc / 32; cb++)
				for (int kr = 0; kr < 32; kr++) {
					int gk = j * Kc + cb * 32 + kr;
					const float *s = J->B + (int64_t)gk * J->sb_k + (int64_t)(c * Nc + ng * 16);
					for (int nr = 0; nr < 16; nr++) {
						int gn = c * Nc + ng * 16 + nr;
						dst[cb * 512 + nr * 32 + kr] = (gk < K && gn < N) ? (f16)(s[nr] * sc) : (f16)0;
					}
				}
		} else {
			for (int cb = 0; cb < Kc / 32; cb++)
				for (int nr = 0; nr < 16; nr++) {
					int gn = c * Nc + ng * 16 + nr;
					f16 *d = dst + cb * 512 + nr * 32;
					int k0 = j * Kc + cb * 32;
					if (gn < N && k0 + 32 <= K) {
						const float *s = J->B + (int64_t)k0 * J->sb_k + (int64_t)gn * J->sb_n;
						if (J->sb_k == 1) { for (int t = 0; t < 32; t++) d[t] = (f16)(s[t] * sc); }
						else { for (int t = 0; t < 32; t++) d[t] = (f16)(s[(int64_t)t * J->sb_k] * sc); }
					} else {
						for (int t = 0; t < 32; t++) {
							int gk = k0 + t;
							d[t] = (gn < N && gk < K) ? (f16)(J->B[(int64_t)gk * J->sb_k + (int64_t)gn * J->sb_n] * sc) : (f16)0;
						}
					}
				}
		}
	}
}

/* C: row-major [Mp][Nc] tiles -> C, reducing K chunks and undoing input scales */
static void unpack_range(void *v, int b, int e) {
	job_t *J = v;
	const int ob = J->dt->out_prec;
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

/* max |A(m,:)| per row (int8) or global max |A| into amax_part[item] */
static void amax_a_rows(void *v, int b, int e) {
	job_t *J = v;
	for (int m = b; m < e; m++) {
		float mx = 0.f;
		for (int k = 0; k < J->K; k++) {
			float x = __builtin_fabsf(J->A[(int64_t)m * J->sa_m + (int64_t)k * J->sa_k]);
			if (x > mx) mx = x;
		}
		J->amax_part[m] = mx;
	}
}
static void amax_b_cols(void *v, int b, int e) {
	job_t *J = v;
	for (int n = b; n < e; n++) {
		float mx = 0.f;
		for (int k = 0; k < J->K; k++) {
			float x = __builtin_fabsf(J->B[(int64_t)k * J->sb_k + (int64_t)n * J->sb_n]);
			if (x > mx) mx = x;
		}
		J->amax_part[n] = mx;
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
 * Engine: one tiled, chained, multi-core NPU matmul
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
	int T = nm * nk * nn;
	ctx->stats.m_tile = Mt; ctx->stats.k_tile = Kc; ctx->stats.n_tile = Nc;
	ctx->stats.tasks += T;

	size_t a_chunk = (size_t)Mp * Kc, b_tile = (size_t)Nc * Kc, c_tile = (size_t)Mp * Nc;
	size_t a_bytes = a_chunk * nk * ib, b_bytes = b_tile * nk * nn * ib, c_bytes = c_tile * nk * nn * ob;
	uint32_t dflags = MEM_FLAG_400 | RKNPU_MEM_CACHEABLE | RKNPU_MEM_NON_CONTIGUOUS;
	if (buf_ensure(ctx, &ctx->a, a_bytes, dflags) ||
	    buf_ensure(ctx, &ctx->b, b_bytes, dflags) ||
	    buf_ensure(ctx, &ctx->c, c_bytes, dflags) ||
	    buf_ensure(ctx, &ctx->regcmd, (size_t)T * REGCMD_SLOTS * 8, dflags) ||
	    buf_ensure(ctx, &ctx->tasks, (size_t)T * sizeof(struct rknpu_task), dflags | RKNPU_MEM_KERNEL_MAPPING))
		return -1;

	J->Mp = Mp; J->Kc = Kc; J->Nc = Nc; J->nk = nk; J->nn = nn;
	J->ap = ctx->a.ptr; J->bp = ctx->b.ptr; J->cp = ctx->c.ptr;
	J->a_chunk = a_chunk; J->b_tile = b_tile; J->c_tile = c_tile;
	J->nblk = ceil_div(Mp, 16);

	double t0 = now_us();
	if (J->A) parallel_for(ctx, nk * J->nblk, 4, pack_a_range, J);
	if (J->B) {
		if (J->cv == CV_F16 && G == 16) parallel_for(ctx, nk * nn * (Nc / 16), 1, pack_b_f16_range, J);
		else parallel_for(ctx, nk * nn * (Nc / G), 1, pack_b_range, J);
	}
	ctx->stats.pack_us += now_us() - t0;
	buf_sync(ctx, &ctx->a, a_bytes, RKNPU_MEM_SYNC_TO_DEVICE);
	buf_sync(ctx, &ctx->b, b_bytes, RKNPU_MEM_SYNC_TO_DEVICE);

	/* register streams: task order (j, c, i) so consecutive tasks share weights */
	uint64_t *rc = ctx->regcmd.ptr;
	struct rknpu_task *tk = ctx->tasks.ptr;
	int cores = ctx->mode == 2 ? 3 : 1;
	int per_submit = ctx->mode == 0 ? 1 : PC_TASK_LIMIT;
	for (int t = 0; t < T; t++) {
		int i = t % nm, jc = t / nm, j = jc / nn;
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
		task_desc d = {
			.dt = dt, .m = Mt, .k = Kc, .n = Nc, .grains = Mt + 1, .dbanks = dbanks, .wbanks = wbanks,
			.reuse = ctx->reuse && weights_fit && !first && i > 0,
			.in_dma = (uint32_t)(ctx->a.dma + ((size_t)j * a_chunk + (size_t)i * Mt * Kc) * ib),
			.w_dma = (uint32_t)(ctx->b.dma + (size_t)jc * b_tile * ib),
			.out_dma = (uint32_t)(ctx->c.dma + ((size_t)jc * c_tile + (size_t)i * Mt * Nc) * ob),
			.next_dma = last ? 0 : (uint32_t)(ctx->regcmd.dma + (size_t)(t + 1) * REGCMD_SLOTS * 8),
		};
		emit_task(rc + (size_t)t * REGCMD_SLOTS, &d);
		struct rknpu_task td = {
			.flags = 0, .op_idx = 0, .enable_mask = 0xd, .int_mask = 0x300, .int_clear = 0x1ffff,
			.int_status = 0, .regcfg_amount = REGCFG_AMOUNT, .regcfg_offset = 0,
			.regcmd_addr = ctx->regcmd.dma + (size_t)t * REGCMD_SLOTS * 8,
		};
		tk[t] = td;
	}
	buf_sync(ctx, &ctx->regcmd, (size_t)T * REGCMD_SLOTS * 8, RKNPU_MEM_SYNC_TO_DEVICE);
	buf_sync(ctx, &ctx->tasks, (size_t)T * sizeof(struct rknpu_task), RKNPU_MEM_SYNC_TO_DEVICE);

	for (int s = 0; s < T; s += per_submit) {
		int cnt = T - s < per_submit ? T - s : per_submit;
		if (submit(ctx, s, cnt, (cores == 3 && cnt >= 3) ? 3 : 1) < 0) return -1;
	}

	buf_sync(ctx, &ctx->c, c_bytes, RKNPU_MEM_SYNC_FROM_DEVICE);
	if (J->C) {
		t0 = now_us();
		parallel_for(ctx, M, 64, unpack_range, J);
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

static float amax_all(rk_ctx *ctx, job_t *J, int rows, range_fn fn) {
	J->amax_part = scratch(ctx, 0, (size_t)rows);
	parallel_for(ctx, rows, 32, fn, J);
	float mx = 0.f;
	for (int i = 0; i < rows; i++) if (J->amax_part[i] > mx) mx = J->amax_part[i];
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

int rk_gemm_ex(rk_ctx *ctx, int prec, int M, int N, int K,
               const float *A, int64_t sa_m, int64_t sa_k,
               const float *B, int64_t sb_k, int64_t sb_n,
               float *C, int64_t ldc, int accumulate) {
	double t_start = now_us();
	memset(&ctx->stats, 0, sizeof(ctx->stats));
	if (M <= 0 || N <= 0 || K <= 0) { set_err(ctx, "bad shape"); return -1; }
	job_t J = {
		.A = A, .B = B, .sa_m = sa_m, .sa_k = sa_k, .sb_k = sb_k, .sb_n = sb_n,
		.M = M, .N = N, .K = K, .C = C, .ldc = ldc, .accumulate = accumulate,
		.a_scale = 1.f, .b_scale = 1.f, .out_mul = 1.f,
	};
	int r = -1;
	double t0;
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
			J.b_scale = pow2_scale(amax_all(ctx, &J, N, amax_b_cols));
			J.out_mul = 1.f / (J.a_scale * J.b_scale);
			ctx->stats.pack_us += now_us() - t0;
		}
		r = gemm_engine(ctx, &J);
		break;
	case RK_PREC_INT8: {
		/* symmetric per-row (A) / per-column (B) quantization, exact int32 accumulation */
		J.dt = &DT_I8_I32;
		J.cv = CV_I8;
		t0 = now_us();
		float *ra = scratch(ctx, 1, (size_t)M), *cb = scratch(ctx, 2, (size_t)N);
		amax_all(ctx, &J, M, amax_a_rows);
		for (int m = 0; m < M; m++) ra[m] = J.amax_part[m] > 0.f ? 127.f / J.amax_part[m] : 1.f;
		amax_all(ctx, &J, N, amax_b_cols);
		for (int n = 0; n < N; n++) cb[n] = J.amax_part[n] > 0.f ? 127.f / J.amax_part[n] : 1.f;
		J.a_row = ra;
		J.b_col = cb;
		ctx->stats.pack_us += now_us() - t0;
		r = gemm_engine(ctx, &J);
		break;
	}
	case RK_PREC_FP16X3: {
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
		/* pass 1: Ah (first K columns of A2) x Bh (rows K..2K-1 of B2) */
		job_t P = J;
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
	ctx->stats.total_us = now_us() - t_start;
	return r;
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
