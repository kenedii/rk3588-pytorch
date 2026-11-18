/*
 * LD_PRELOAD shim that logs RKNPU ioctls made by any program (e.g. librknnrt) and
 * dumps the register command streams of every submitted task.
 *
 *   gcc -O2 -shared -fPIC -o ioctl_spy.so tools/ioctl_spy.c -ldl
 *   LD_PRELOAD=./ioctl_spy.so SPY_OUT=/tmp/spy.txt ./program
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <pthread.h>
#include "../csrc/rknpu_uapi.h"

#define MAXB 4096
typedef struct { uint32_t handle; uint64_t obj, dma, size, offset; uint32_t flags, core_mask; int32_t dom; void *ptr; } ent;
static ent tab[MAXB];
static int ntab;
static FILE *out;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static int (*real_ioctl)(int, unsigned long, ...);
static void *(*real_mmap)(void *, size_t, int, int, int, off_t);

static void init(void) {
	if (!real_ioctl) real_ioctl = dlsym(RTLD_NEXT, "ioctl");
	if (!real_mmap) real_mmap = dlsym(RTLD_NEXT, "mmap");
	if (!out) { const char *p = getenv("SPY_OUT"); out = p ? fopen(p, "w") : stderr; if (!out) out = stderr; }
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
	init();
	void *p = real_mmap(addr, len, prot, flags, fd, off);
	pthread_mutex_lock(&mu);
	for (int i = 0; i < ntab; i++) if (tab[i].offset == (uint64_t)off && off) tab[i].ptr = p;
	pthread_mutex_unlock(&mu);
	return p;
}

void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off_t off) { return mmap(addr, len, prot, flags, fd, off); }

static ent *by_dma(uint64_t dma) {
	for (int i = ntab - 1; i >= 0; i--) if (dma >= tab[i].dma && dma < tab[i].dma + tab[i].size) return &tab[i];
	return NULL;
}
static ent *by_obj(uint64_t obj) {
	for (int i = ntab - 1; i >= 0; i--) if (tab[i].obj == obj) return &tab[i];
	return NULL;
}

int ioctl(int fd, unsigned long req, ...) {
	va_list ap; va_start(ap, req); void *arg = va_arg(ap, void *); va_end(ap);
	init();
	if (req == DRM_IOCTL_RKNPU_SUBMIT) {
		struct rknpu_submit *s = arg;
		pthread_mutex_lock(&mu);
		fprintf(out, "SUBMIT flags=%#x timeout=%u start=%u num=%u prio=%d task_obj=%#llx dom=%u base=%#llx core_mask=%#x fence=%d sub=[%u,%u][%u,%u][%u,%u][%u,%u][%u,%u]\n",
			s->flags, s->timeout, s->task_start, s->task_number, s->priority, (unsigned long long)s->task_obj_addr,
			s->iommu_domain_id, (unsigned long long)s->task_base_addr, s->core_mask, s->fence_fd,
			s->subcore_task[0].task_start, s->subcore_task[0].task_number, s->subcore_task[1].task_start, s->subcore_task[1].task_number,
			s->subcore_task[2].task_start, s->subcore_task[2].task_number, s->subcore_task[3].task_start, s->subcore_task[3].task_number,
			s->subcore_task[4].task_start, s->subcore_task[4].task_number);
		ent *t = by_obj(s->task_obj_addr);
		int maxdump = getenv("SPY_MAXTASKS") ? atoi(getenv("SPY_MAXTASKS")) : 4;
		if (t && t->ptr) {
			struct rknpu_task *tk = t->ptr;
			for (uint32_t i = s->task_start; i < s->task_start + s->task_number && (int)(i - s->task_start) < maxdump; i++) {
				fprintf(out, " TASK %u flags=%#x op_idx=%u enable=%#x int_mask=%#x int_clear=%#x amount=%u roff=%u regcmd=%#llx\n",
					i, tk[i].flags, tk[i].op_idx, tk[i].enable_mask, tk[i].int_mask, tk[i].int_clear,
					tk[i].regcfg_amount, tk[i].regcfg_offset, (unsigned long long)tk[i].regcmd_addr);
				ent *r = by_dma(tk[i].regcmd_addr);
				if (r && r->ptr) {
					uint64_t *ops = (uint64_t *)((char *)r->ptr + (tk[i].regcmd_addr - r->dma));
					for (uint32_t k = 0; k < tk[i].regcfg_amount + 4 && k < 512; k++)
						fprintf(out, "  %3u op=%04x reg=%04x val=%08x\n", k, (unsigned)(ops[k] >> 48),
							(unsigned)(ops[k] & 0xffff), (unsigned)((ops[k] >> 16) & 0xffffffff));
					if (getenv("SPY_DUMP_W") && i == s->task_start) {
						/* dump the first fp16 values of the weight buffer (reg 0x1110) */
						for (uint32_t k = 0; k < tk[i].regcfg_amount; k++) {
							if ((ops[k] & 0xffff) != 0x1110) continue;
							uint32_t wd = (uint32_t)((ops[k] >> 16) & 0xffffffff);
							ent *w = by_dma(wd);
							if (w && w->ptr) {
								_Float16 *h = (_Float16 *)((char *)w->ptr + (wd - w->dma));
								signed char *hb = (signed char *)((char *)w->ptr + (wd - w->dma));
								int nd = atoi(getenv("SPY_DUMP_W"));
								int bytes = getenv("SPY_W_BYTES") != NULL;
								fprintf(out, "  WEIGHTS@%#x:", wd);
								for (int q = 0; q < nd; q++) fprintf(out, "%s%d", q % 32 ? " " : "\n   ", bytes ? (int)hb[q] : (int)h[q]);
								fprintf(out, "\n");
							}
							break;
						}
					}
				} else fprintf(out, "  (regcmd buffer not mapped by this process)\n");
			}
		} else fprintf(out, " (task buffer not mapped)\n");
		fflush(out);
		pthread_mutex_unlock(&mu);
	}
	int r = real_ioctl(fd, req, arg);
	if (req == DRM_IOCTL_RKNPU_MEM_CREATE) {
		struct rknpu_mem_create *c = arg;
		pthread_mutex_lock(&mu);
		fprintf(out, "MEM_CREATE ret=%d flags=%#x size=%llu dom=%d core_mask=%#x -> handle=%u obj=%#llx dma=%#llx\n", r, c->flags,
			(unsigned long long)c->size, c->iommu_domain_id, c->core_mask, c->handle, (unsigned long long)c->obj_addr, (unsigned long long)c->dma_addr);
		if (!r && ntab < MAXB) tab[ntab++] = (ent){ c->handle, c->obj_addr, c->dma_addr, c->size, 0, c->flags, c->core_mask, c->iommu_domain_id, NULL };
		fflush(out);
		pthread_mutex_unlock(&mu);
	} else if (req == DRM_IOCTL_RKNPU_MEM_MAP) {
		struct rknpu_mem_map *m = arg;
		pthread_mutex_lock(&mu);
		for (int i = 0; i < ntab; i++) if (tab[i].handle == m->handle) tab[i].offset = m->offset;
		pthread_mutex_unlock(&mu);
	} else if (req == DRM_IOCTL_RKNPU_ACTION) {
		struct rknpu_action *a = arg;
		fprintf(out, "ACTION flags=%u value=%#x ret=%d\n", a->flags, a->value, r);
	} else if (req == DRM_IOCTL_RKNPU_MEM_SYNC) {
		struct rknpu_mem_sync *s = arg;
		fprintf(out, "MEM_SYNC flags=%#x obj=%#llx off=%llu size=%llu ret=%d\n", s->flags, (unsigned long long)s->obj_addr,
			(unsigned long long)s->offset, (unsigned long long)s->size, r);
	} else if ((req & 0xff00) == 0x6400 && getenv("SPY_ALL")) {
		fprintf(out, "IOCTL %#lx ret=%d\n", req, r);
	}
	return r;
}
