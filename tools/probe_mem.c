/* Probe which RKNPU allocation flags can be mmapped and touched from userspace. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <setjmp.h>
#include <errno.h>
#include <sys/mman.h>
#include "../csrc/rknpu_uapi.h"

static sigjmp_buf jb;
static void on_bus(int s) { siglongjmp(jb, s); }

int main(int argc, char **argv) {
	const char *dev = argc > 1 ? argv[1] : "/dev/dri/card2";
	int fd = open(dev, O_RDWR | O_CLOEXEC);
	if (fd < 0) { perror("open"); return 1; }
	struct rknpu_action a = { .flags = RKNPU_GET_DRV_VERSION };
	ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &a); printf("drv version %u\n", a.value);
	a.flags = RKNPU_GET_HW_VERSION; a.value = 0; ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &a); printf("hw version 0x%x\n", a.value);
	a.flags = RKNPU_GET_IOMMU_EN; a.value = 0; ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &a); printf("iommu en %u\n", a.value);
	a.flags = RKNPU_GET_FREQ; a.value = 0; ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &a); printf("freq %u\n", a.value);
	signal(SIGBUS, on_bus); signal(SIGSEGV, on_bus);
	uint32_t flags[] = {0, RKNPU_MEM_KERNEL_MAPPING, RKNPU_MEM_IOMMU, RKNPU_MEM_NON_CONTIGUOUS,
		RKNPU_MEM_NON_CONTIGUOUS|RKNPU_MEM_IOMMU, RKNPU_MEM_CACHEABLE, RKNPU_MEM_CACHEABLE|RKNPU_MEM_IOMMU,
		RKNPU_MEM_NON_CONTIGUOUS|RKNPU_MEM_CACHEABLE, RKNPU_MEM_NON_CONTIGUOUS|RKNPU_MEM_CACHEABLE|RKNPU_MEM_IOMMU,
		RKNPU_MEM_WRITE_COMBINE|RKNPU_MEM_IOMMU, RKNPU_MEM_NON_CONTIGUOUS|RKNPU_MEM_WRITE_COMBINE|RKNPU_MEM_IOMMU,
		RKNPU_MEM_CACHEABLE|RKNPU_MEM_IOMMU|RKNPU_MEM_KERNEL_MAPPING, RKNPU_MEM_NON_CONTIGUOUS|RKNPU_MEM_CACHEABLE|RKNPU_MEM_IOMMU|RKNPU_MEM_KERNEL_MAPPING};
	size_t sizes[] = {4096, 1 << 20};
	for (unsigned i = 0; i < sizeof(flags)/sizeof(flags[0]); i++) for (int s = 0; s < 2; s++) {
		struct rknpu_mem_create mc = { .flags = flags[i], .size = sizes[s], .core_mask = 7 };
		int r = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mc);
		if (r) { printf("flags 0x%02x size %zu: create failed errno %d\n", flags[i], sizes[s], errno); continue; }
		struct rknpu_mem_map mm = { .handle = mc.handle };
		r = ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &mm);
		void *p = mmap(NULL, sizes[s], PROT_READ|PROT_WRITE, MAP_SHARED, fd, mm.offset);
		const char *st = "ok";
		if (r || p == MAP_FAILED) st = "map failed";
		else if (sigsetjmp(jb, 1)) st = "FAULT on touch";
		else { memset(p, 0x5a, sizes[s]); volatile unsigned char c = ((unsigned char*)p)[sizes[s]-1]; (void)c; }
		printf("flags 0x%02x size %7zu: dma 0x%08llx obj 0x%llx off 0x%llx -> %s\n", flags[i], sizes[s],
		       (unsigned long long)mc.dma_addr, (unsigned long long)mc.obj_addr, (unsigned long long)mm.offset, st);
		if (p != MAP_FAILED) munmap(p, sizes[s]);
		struct rknpu_mem_destroy md = { .handle = mc.handle, .obj_addr = mc.obj_addr };
		ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &md);
	}
	close(fd);
	return 0;
}
