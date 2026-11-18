/*
 * Minimal userspace view of the Rockchip RKNPU DRM ioctl interface
 * (vendor driver 0.9.x as found in rknpu 0.9.8). Only what rk3588-pytorch uses.
 */
#ifndef RK3588_PYTORCH_RKNPU_UAPI_H
#define RK3588_PYTORCH_RKNPU_UAPI_H

#include <stdint.h>
#include <sys/ioctl.h>

/* rknpu_mem_create.flags */
#define RKNPU_MEM_CONTIGUOUS       0u
#define RKNPU_MEM_NON_CONTIGUOUS   (1u << 0)
#define RKNPU_MEM_CACHEABLE        (1u << 1)
#define RKNPU_MEM_WRITE_COMBINE    (1u << 2)
#define RKNPU_MEM_KERNEL_MAPPING   (1u << 3)
#define RKNPU_MEM_IOMMU            (1u << 4)
#define RKNPU_MEM_ZEROING          (1u << 5)
#define RKNPU_MEM_NON_DMA32        (1u << 7)

/* rknpu_mem_sync.flags */
#define RKNPU_MEM_SYNC_TO_DEVICE   (1u << 0)
#define RKNPU_MEM_SYNC_FROM_DEVICE (1u << 1)

/* rknpu_submit.flags */
#define RKNPU_JOB_PC               (1u << 0)
#define RKNPU_JOB_NONBLOCK         (1u << 1)
#define RKNPU_JOB_PINGPONG         (1u << 2)

/* rknpu_action.flags */
#define RKNPU_GET_HW_VERSION       0
#define RKNPU_GET_DRV_VERSION      1
#define RKNPU_GET_FREQ             2
#define RKNPU_GET_IOMMU_EN         18

struct rknpu_mem_create {
	uint32_t handle;
	uint32_t flags;
	uint64_t size;
	uint64_t obj_addr;
	uint64_t dma_addr;
	uint64_t sram_size;
	int32_t  iommu_domain_id;
	uint32_t core_mask;
};

struct rknpu_mem_map {
	uint32_t handle;
	uint32_t reserved;
	uint64_t offset;
};

struct rknpu_mem_destroy {
	uint32_t handle;
	uint32_t reserved;
	uint64_t obj_addr;
};

struct rknpu_mem_sync {
	uint32_t flags;
	uint32_t reserved;
	uint64_t obj_addr;
	uint64_t offset;
	uint64_t size;
};

struct rknpu_task {
	uint32_t flags;
	uint32_t op_idx;
	uint32_t enable_mask;
	uint32_t int_mask;
	uint32_t int_clear;
	uint32_t int_status;
	uint32_t regcfg_amount;
	uint32_t regcfg_offset;
	uint64_t regcmd_addr;
} __attribute__((packed));

struct rknpu_subcore_task {
	uint32_t task_start;
	uint32_t task_number;
};

struct rknpu_submit {
	uint32_t flags;
	uint32_t timeout;
	uint32_t task_start;
	uint32_t task_number;
	uint32_t task_counter;
	int32_t  priority;
	uint64_t task_obj_addr;
	uint32_t iommu_domain_id;
	uint32_t reserved;
	uint64_t task_base_addr;
	int64_t  hw_elapse_time;
	uint32_t core_mask;
	int32_t  fence_fd;
	struct rknpu_subcore_task subcore_task[5];
};

struct rknpu_action {
	uint32_t flags;
	uint32_t value;
};

/* DRM ioctl encoding: DRM_IOWR(DRM_COMMAND_BASE + nr, type), DRM_IOCTL_BASE 'd', DRM_COMMAND_BASE 0x40 */
#define RKNPU_DRM_IOWR(nr, type) _IOWR('d', 0x40 + (nr), type)
#define DRM_IOCTL_RKNPU_ACTION      RKNPU_DRM_IOWR(0x00, struct rknpu_action)
#define DRM_IOCTL_RKNPU_SUBMIT      RKNPU_DRM_IOWR(0x01, struct rknpu_submit)
#define DRM_IOCTL_RKNPU_MEM_CREATE  RKNPU_DRM_IOWR(0x02, struct rknpu_mem_create)
#define DRM_IOCTL_RKNPU_MEM_MAP     RKNPU_DRM_IOWR(0x03, struct rknpu_mem_map)
#define DRM_IOCTL_RKNPU_MEM_DESTROY RKNPU_DRM_IOWR(0x04, struct rknpu_mem_destroy)
#define DRM_IOCTL_RKNPU_MEM_SYNC    RKNPU_DRM_IOWR(0x05, struct rknpu_mem_sync)

#endif
