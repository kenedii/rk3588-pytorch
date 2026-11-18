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

typedef struct {
	int m, k, n;            /* rows in this task, channels, kernels */
	int grains;             /* feature rows the CNA preloads */
	int dbanks, wbanks;     /* CBUF split */
	int reuse;              /* weights already resident from the previous task */
	uint32_t in_dma, w_dma, out_dma, next_dma;
} task_desc;

/* Register stream for one row-major fp16 x fp16 -> fp32 matmul task. */
static void emit_task(uint64_t *o, const task_desc *t) {
	const uint32_t m = t->m, k = t->k, n = t->n;
	const uint32_t cbuf = (t->reuse ? CBUF_WEIGHT_REUSE : 0) | ((uint32_t)t->wbanks << 4) | (uint32_t)t->dbanks;
	const uint32_t conv1 = (1u << 29) | (PREC_FP16 << 7) | (PREC_FP16 << 4);  /* bit 29: row-major feature input */
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
	R(OP_REG_CNA, CNA_WEIGHT_SIZE0, k * 2u * n);
	R(OP_REG_CNA, CNA_WEIGHT_SIZE1, (k * 2u) & 0x7FFFF);
	R(OP_REG_CNA, CNA_WEIGHT_SIZE2, (1u << 24) | (1u << 16) | (n & 0x3FFF));
	R(OP_REG_CNA, CNA_CBUF_CON0, cbuf);
	R(OP_REG_CNA, CNA_CBUF_CON1, ((k + 31) / 32) & 0x1FFF);
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
	R(OP_REG_CNA, CNA_DMA_CON1, k / 8);          /* line stride: one row of A, in 16-byte units */
	R(OP_REG_CNA, CNA_DMA_CON2, 0);              /* surface stride: channel groups are contiguous */
	R(OP_REG_CNA, CNA_FC_DATA_SIZE0, (1u << 16) | (m & 0x7FF));
	R(OP_REG_CNA, CNA_FC_DATA_SIZE1, k & 0xFFFF);
	R(OP_REG_CNA, CNA_DCOMP_CTRL, 0);
	R(OP_REG_CNA, CNA_DCOMP_REGNUM, 0);
	R(OP_REG_CNA, CNA_DCOMP_ADDR0, t->w_dma);
	for (int r = 0; r < 16; r++) R(OP_REG_CNA, CNA_DCOMP_AMOUNT0 + 4 * r, 0);
	R(OP_REG_CNA, CNA_CVT_CON5, 0);
	R(OP_REG_CNA, CNA_PAD_CON1, 0);

	R(OP_REG_CORE, CORE_MISC_CFG, (PREC_FP16 << 8) | 1);
	R(OP_REG_CORE, CORE_DATAOUT_SIZE_0, (m - 1) << 16);
	R(OP_REG_CORE, CORE_DATAOUT_SIZE_1, n - 1);
	R(OP_REG_CORE, CORE_CLIP_TRUNCATE, 0);
	R(OP_REG_CORE, CORE_3030, 0);

	R(OP_REG_DPU, DPU_FEATURE_MODE_CFG, (15u << 5) | (2u << 1));
	R(OP_REG_DPU, DPU_DATA_FORMAT, ((uint32_t)PREC_FP32 << 29) | (PREC_FP16 << 26) | PREC_FP16);
	R(OP_REG_DPU, DPU_OFFSET_PEND, 0);
	R(OP_REG_DPU, DPU_DST_BASE_ADD, t->out_dma);
	R(OP_REG_DPU, DPU_DST_SURF_STRIDE, 1u << 4);  /* row-major output: 4-float atoms are adjacent */
	R(OP_REG_DPU, DPU_DATA_CUBE_WIDTH, 0);
	R(OP_REG_DPU, DPU_DATA_CUBE_HEIGHT, (m - 1) & 0x1FFF);
	R(OP_REG_DPU, DPU_DATA_CUBE_NOTCH_ADDR, ((n / 4 - 1) << 16) | (n / 4 - 1));
	R(OP_REG_DPU, DPU_DATA_CUBE_CHANNEL, ((n - 1) << 16) | (n - 1));
	R(OP_REG_DPU, DPU_BS_CFG, (1 << 6) | (1 << 4) | (1 << 1) | 1);
	R(OP_REG_DPU, DPU_BS_ALU_CFG, 0);
	R(OP_REG_DPU, DPU_BS_MUL_CFG, 0);
	R(OP_REG_DPU, DPU_BS_RELUX_CMP_VALUE, 0);
	R(OP_REG_DPU, DPU_BS_OW_CFG, (3 << 8) | (3 << 5) | (3 << 2) | (1 << 1));
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
	R(OP_REG_DPU, DPU_OUT_CVT_SCALE, 1);
	R(OP_REG_DPU, DPU_OUT_CVT_SHIFT, 0);
	for (int r = 0; r < 8; r++) R(OP_REG_DPU, DPU_EW_OP_VALUE_0 + 4 * r, 0);
	R(OP_REG_DPU, DPU_SURFACE_ADD, 4u << 4);
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

/* Convert rows [r0, r1) of op(A) (fp32, strided) into a row-major fp16 [rows][Kc] chunk. */
static void pack_a_rows(f16 *dst, const float *A, int64_t sa_m, int64_t sa_k, int M, int K,
                        int r0, int r1, int k0, int Kc) {
	for (int m = r0; m < r1; m++) {
		f16 *d = dst + (size_t)m * Kc;
		int kend = K - k0 < Kc ? K - k0 : Kc;
		if (m >= M) { memset(d, 0, (size_t)Kc * 2); continue; }
		if (kend < 0) kend = 0;
		const float *s = A + (int64_t)m * sa_m + (int64_t)k0 * sa_k;
		if (sa_k == 1) { for (int k = 0; k < kend; k++) d[k] = (f16)s[k]; }
		else { for (int k = 0; k < kend; k++) d[k] = (f16)s[(int64_t)k * sa_k]; }
		for (int k = kend; k < Kc; k++) d[k] = (f16)0;
	}
}

/* Transposed source (sa_m == 1): walk in 16x64 blocks to keep reads and writes cached. */
static void pack_a_blocked(f16 *dst, const float *A, int64_t sa_k, int M, int K, int r0, int r1, int k0, int Kc) {
	for (int mb = r0; mb < r1; mb += 16) {
		int me = mb + 16 < r1 ? mb + 16 : r1;
		for (int kb = 0; kb < Kc; kb += 64) {
			int ke = kb + 64 < Kc ? kb + 64 : Kc;
			for (int k = kb; k < ke; k++) {
				int gk = k0 + k;
				const float *s = A + (int64_t)gk * sa_k;
				for (int m = mb; m < me; m++)
					dst[(size_t)m * Kc + k] = (m < M && gk < K) ? (f16)s[m] : (f16)0;
			}
		}
	}
}

typedef struct {
	const float *A, *B;
	int64_t sa_m, sa_k, sb_k, sb_n;
	int M, N, K, Mp, Kc, Nc, nk, nn, nblk;
	f16 *ap, *bp;
	float *cp;
	size_t a_chunk, b_tile, c_tile;
	float *C;
	int64_t ldc;
	int accumulate;
} pack_job;

/* A: fp32 -> fp16, row-major [Mp][Kc] per K chunk; items are (K chunk, 16-row block) */
static void pack_a_range(void *v, int b, int e) {
	pack_job *J = v;
	for (int it = b; it < e; it++) {
		int j = it / J->nblk, bk = it % J->nblk;
		int r0 = bk * 16, r1 = r0 + 16 < J->Mp ? r0 + 16 : J->Mp;
		if (J->sa_k == 1 || J->sa_m != 1)
			pack_a_rows(J->ap + j * J->a_chunk, J->A, J->sa_m, J->sa_k, J->M, J->K, r0, r1, j * J->Kc, J->Kc);
		else
			pack_a_blocked(J->ap + j * J->a_chunk, J->A, J->sa_k, J->M, J->K, r0, r1, j * J->Kc, J->Kc);
	}
}

/* B: kernel groups [Nc/16][Kc/32][16][32] per (K chunk, N chunk); items are 16-kernel groups */
static void pack_b_range(void *v, int b, int e) {
	pack_job *J = v;
	const int Kc = J->Kc, Nc = J->Nc, K = J->K, N = J->N;
	const float *B = J->B;
	const int64_t sb_k = J->sb_k, sb_n = J->sb_n;
	for (int it = b; it < e; it++) {
		int jc = it / (Nc / 16), ng = it % (Nc / 16);
		int j = jc / J->nn, c = jc % J->nn;
		f16 *dst = J->bp + (size_t)jc * J->b_tile + (size_t)ng * 16 * Kc;
		if (sb_n == 1 && sb_k != 1) {
			/* B row-major (k, n): read 16 consecutive n per k */
			for (int cb = 0; cb < Kc / 32; cb++)
				for (int kr = 0; kr < 32; kr++) {
					int gk = j * Kc + cb * 32 + kr;
					const float *s = B + (int64_t)gk * sb_k + (int64_t)(c * Nc + ng * 16);
					for (int nr = 0; nr < 16; nr++) {
						int gn = c * Nc + ng * 16 + nr;
						dst[cb * 512 + nr * 32 + kr] = (gk < K && gn < N) ? (f16)s[nr] : (f16)0;
					}
				}
		} else {
			for (int cb = 0; cb < Kc / 32; cb++) {
				for (int nr = 0; nr < 16; nr++) {
					int gn = c * Nc + ng * 16 + nr;
					f16 *d = dst + cb * 512 + nr * 32;
					int k0 = j * Kc + cb * 32;
					if (gn < N && k0 + 32 <= K) {
						const float *s = B + (int64_t)k0 * sb_k + (int64_t)gn * sb_n;
						if (sb_k == 1) { for (int t = 0; t < 32; t++) d[t] = (f16)s[t]; }
						else { for (int t = 0; t < 32; t++) d[t] = (f16)s[(int64_t)t * sb_k]; }
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
}

/* C: copy (and reduce over K chunks) rows of the row-major [Mp][Nc] output tiles */
static void unpack_range(void *v, int b, int e) {
	pack_job *J = v;
	for (int m = b; m < e; m++) {
		float *crow = J->C + (int64_t)m * J->ldc;
		for (int c = 0; c < J->nn; c++) {
			int n0 = c * J->Nc;
			int ncols = J->N - n0 < J->Nc ? J->N - n0 : J->Nc;
			const float *src = J->cp + (size_t)c * J->c_tile + (size_t)m * J->Nc;
			if (J->nk == 1 && !J->accumulate) { memcpy(crow + n0, src, (size_t)ncols * 4); continue; }
			for (int n = 0; n < ncols; n++) {
				float acc = 0.f;
				for (int j = 0; j < J->nk; j++)
					acc += J->cp[(size_t)(j * J->nn + c) * J->c_tile + (size_t)m * J->Nc + n];
				if (J->accumulate) crow[n0 + n] += acc; else crow[n0 + n] = acc;
			}
		}
	}
}

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
	int Mp = round_up(M, 4);
	/* weight banks: at least one 16-kernel group, and (like librknnrt) up to 4 banks or
	 * whatever holds all weights; leaving a single bank for streamed weights hangs the CNA */
	int wbanks_min = ceil_div(16 * Kc * 2, CBUF_BANK_SIZE);
	int wall = ceil_div(Kc * 2 * Nc, CBUF_BANK_SIZE);
	int wfloor = env_int("RKNPU_MIN_WBANKS", 4);
	if (wall < wfloor) wfloor = wall;
	if (wbanks_min < wfloor) wbanks_min = wfloor;
	int dbanks_max = CBUF_BANKS - wbanks_min;
	if (dbanks_max < 1) { set_err(ctx, "K chunk %d too large for CBUF", Kc); return -1; }
	/* rows per task: (rows + 1) must fit in the data banks so the CNA can preload rows + 1 */
	int Mt = (int)((dbanks_max * (long)CBUF_BANK_SIZE) / (Kc * 2)) - 1;
	Mt &= ~3;
	if (Mt > ctx->max_m) Mt = ctx->max_m & ~3;
	if (Mt > Mp) Mt = Mp;
	if (Mt < 4) { set_err(ctx, "K chunk %d leaves no room for 4 rows", Kc); return -1; }
	int nm = ceil_div(Mp, Mt);
	Mt = round_up(ceil_div(Mp, nm), 4);   /* rebalance */
	Mp = Mt * nm;
	int dbanks = ceil_div((Mt + 1) * Kc * 2, CBUF_BANK_SIZE);
	int wbanks = CBUF_BANKS - dbanks;
	int weights_fit = (long)Kc * 2 * Nc <= (long)wbanks * CBUF_BANK_SIZE;
	int T = nm * nk * nn;
	ctx->stats.m_tile = Mt; ctx->stats.k_tile = Kc; ctx->stats.n_tile = Nc;
	ctx->stats.tasks = T;

	size_t a_chunk = (size_t)Mp * Kc;   /* fp16 elements per K chunk */
	size_t b_tile = (size_t)Nc * Kc;    /* fp16 elements per (K chunk, N chunk) */
	size_t c_tile = (size_t)Mp * Nc;    /* fp32 elements per (K chunk, N chunk) */
	size_t a_bytes = a_chunk * nk * 2, b_bytes = b_tile * nk * nn * 2, c_bytes = c_tile * nk * nn * 4;
	uint32_t dflags = MEM_FLAG_400 | RKNPU_MEM_CACHEABLE | RKNPU_MEM_NON_CONTIGUOUS;
	if (buf_ensure(ctx, &ctx->a, a_bytes, dflags) ||
	    buf_ensure(ctx, &ctx->b, b_bytes, dflags) ||
	    buf_ensure(ctx, &ctx->c, c_bytes, dflags) ||
	    buf_ensure(ctx, &ctx->regcmd, (size_t)T * REGCMD_SLOTS * 8, dflags) ||
	    buf_ensure(ctx, &ctx->tasks, (size_t)T * sizeof(struct rknpu_task), dflags | RKNPU_MEM_KERNEL_MAPPING))
		return -1;

	pack_job job = {
		.A = A, .B = B, .sa_m = sa_m, .sa_k = sa_k, .sb_k = sb_k, .sb_n = sb_n,
		.M = M, .N = N, .K = K, .Mp = Mp, .Kc = Kc, .Nc = Nc, .nk = nk, .nn = nn,
		.ap = ctx->a.ptr, .bp = ctx->b.ptr, .cp = ctx->c.ptr, .a_chunk = a_chunk, .b_tile = b_tile,
		.c_tile = c_tile, .C = C, .ldc = ldc, .accumulate = accumulate,
	};
	job.nblk = ceil_div(Mp, 16);
	double t0 = now_us();
	parallel_for(ctx, nk * job.nblk, 4, pack_a_range, &job);
	parallel_for(ctx, nk * nn * (Nc / 16), 1, pack_b_range, &job);
	ctx->stats.pack_us = now_us() - t0;
	buf_sync(ctx, &ctx->a, a_bytes, RKNPU_MEM_SYNC_TO_DEVICE);
	buf_sync(ctx, &ctx->b, b_bytes, RKNPU_MEM_SYNC_TO_DEVICE);

	/* ---- register streams: task order (j, c, i) so consecutive tasks share weights ---- */
	uint64_t *rc = ctx->regcmd.ptr;
	struct rknpu_task *tk = ctx->tasks.ptr;
	int cores = ctx->mode == 2 ? 3 : 1;
	int per_submit = ctx->mode == 0 ? 1 : PC_TASK_LIMIT;
	for (int t = 0; t < T; t++) {
		int i = t % nm, jc = t / nm, j = jc / nn;
		int sub_start = (t / per_submit) * per_submit;
		int sub_count = T - sub_start < per_submit ? T - sub_start : per_submit;
		/* first/last task of each chain (one chain per submit, or per core in mode 2) */
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
			.m = Mt, .k = Kc, .n = Nc, .grains = Mt + 1, .dbanks = dbanks, .wbanks = wbanks,
			.reuse = ctx->reuse && weights_fit && !first && i > 0,
			.in_dma = (uint32_t)(ctx->a.dma + ((size_t)j * a_chunk + (size_t)i * Mt * Kc) * 2),
			.w_dma = (uint32_t)(ctx->b.dma + (size_t)jc * b_tile * 2),
			.out_dma = (uint32_t)(ctx->c.dma + ((size_t)jc * c_tile + (size_t)i * Mt * Nc) * 4),
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

	/* ---- run ---- */
	for (int s = 0; s < T; s += per_submit) {
		int cnt = T - s < per_submit ? T - s : per_submit;
		if (submit(ctx, s, cnt, (cores == 3 && cnt >= 3) ? 3 : 1) < 0) return -1;
	}

	/* ---- C: copy (and reduce over K chunks) from row-major [Mp][Nc] tiles ---- */
	buf_sync(ctx, &ctx->c, c_bytes, RKNPU_MEM_SYNC_FROM_DEVICE);
	t0 = now_us();
	parallel_for(ctx, M, 64, unpack_range, &job);
	ctx->stats.unpack_us = now_us() - t0;
	ctx->stats.total_us = now_us() - t_start;
	return 0;
}
