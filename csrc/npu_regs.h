/*
 * RK3588 NPU register offsets and register-command encoding.
 *
 * Offsets follow the RK3588 TRM (v1.0) naming. The command stream encoding and the
 * register sequence for a CNA->CORE->DPU convolution were reverse engineered by
 * Jasbir Matharu (https://github.com/mtx512/rk3588-npu, GPL-3.0).
 */
#ifndef RK3588_PYTORCH_NPU_REGS_H
#define RK3588_PYTORCH_NPU_REGS_H

#include <stdint.h>

#define PC_OPERATION_ENABLE    0x0008
#define PC_BASE_ADDRESS        0x0010
#define PC_REGISTER_AMOUNTS    0x0014

#define CNA_CONV_CON1          0x100C
#define CNA_CONV_CON2          0x1010
#define CNA_CONV_CON3          0x1014
#define CNA_DATA_SIZE0         0x1020
#define CNA_DATA_SIZE1         0x1024
#define CNA_DATA_SIZE2         0x1028
#define CNA_DATA_SIZE3         0x102C
#define CNA_WEIGHT_SIZE0       0x1030
#define CNA_WEIGHT_SIZE1       0x1034
#define CNA_WEIGHT_SIZE2       0x1038
#define CNA_CBUF_CON0          0x1040
#define CNA_CBUF_CON1          0x1044
#define CNA_CVT_CON0           0x104C
#define CNA_CVT_CON1           0x1050
#define CNA_CVT_CON2           0x1054
#define CNA_CVT_CON3           0x1058
#define CNA_CVT_CON4           0x105C
#define CNA_FC_CON0            0x1060
#define CNA_FC_CON1            0x1064
#define CNA_PAD_CON0           0x1068
#define CNA_FEATURE_DATA_ADDR  0x1070
#define CNA_FC_CON2            0x1074
#define CNA_DMA_CON0           0x1078
#define CNA_DMA_CON1           0x107C
#define CNA_DMA_CON2           0x1080
#define CNA_FC_DATA_SIZE0      0x1084
#define CNA_FC_DATA_SIZE1      0x1088
#define CNA_DCOMP_CTRL         0x1100
#define CNA_DCOMP_REGNUM       0x1104
#define CNA_DCOMP_ADDR0        0x1110
#define CNA_DCOMP_AMOUNT0      0x1140 /* 16 consecutive registers */
#define CNA_CVT_CON5           0x1180
#define CNA_PAD_CON1           0x1184

#define CORE_MISC_CFG          0x3010
#define CORE_DATAOUT_SIZE_0    0x3014
#define CORE_DATAOUT_SIZE_1    0x3018
#define CORE_CLIP_TRUNCATE     0x301C
#define CORE_3030              0x3030

#define DPU_S_POINTER            0x4004
#define DPU_FEATURE_MODE_CFG     0x400C
#define DPU_DATA_FORMAT          0x4010
#define DPU_OFFSET_PEND          0x4014
#define DPU_DST_BASE_ADD         0x4020
#define DPU_DST_SURF_STRIDE      0x4024
#define DPU_DATA_CUBE_WIDTH      0x4030
#define DPU_DATA_CUBE_HEIGHT     0x4034
#define DPU_DATA_CUBE_NOTCH_ADDR 0x4038
#define DPU_DATA_CUBE_CHANNEL    0x403C
#define DPU_BS_CFG               0x4040
#define DPU_BS_ALU_CFG           0x4044
#define DPU_BS_MUL_CFG           0x4048
#define DPU_BS_RELUX_CMP_VALUE   0x404C
#define DPU_BS_OW_CFG            0x4050
#define DPU_BS_OW_OP             0x4054
#define DPU_WDMA_SIZE_0          0x4058
#define DPU_WDMA_SIZE_1          0x405C
#define DPU_BN_CFG               0x4060
#define DPU_BN_ALU_CFG           0x4064
#define DPU_BN_MUL_CFG           0x4068
#define DPU_BN_RELUX_CMP_VALUE   0x406C
#define DPU_EW_CFG               0x4070
#define DPU_EW_CVT_OFFSET_VALUE  0x4074
#define DPU_EW_CVT_SCALE_VALUE   0x4078
#define DPU_EW_RELUX_CMP_VALUE   0x407C
#define DPU_OUT_CVT_OFFSET       0x4080
#define DPU_OUT_CVT_SCALE        0x4084
#define DPU_OUT_CVT_SHIFT        0x4088
#define DPU_EW_OP_VALUE_0        0x4090 /* 8 consecutive registers */
#define DPU_SURFACE_ADD          0x40C0
#define DPU_40C4                 0x40C4
#define DPU_LUT_ACCESS_CFG       0x4100 /* 12 consecutive LUT registers up to 0x412C */

/* Target block selectors (upper 16 bits of a command word) */
#define BLK_PC       0x0100
#define BLK_CNA      0x0200
#define BLK_CORE     0x0800
#define BLK_DPU      0x1000
#define OP_REG       0x0001
#define OP_REG_PC    (BLK_PC | OP_REG)
#define OP_REG_CNA   (BLK_CNA | OP_REG)
#define OP_REG_CORE  (BLK_CORE | OP_REG)
#define OP_REG_DPU   (BLK_DPU | OP_REG)
#define OP_40        (0x40 | OP_REG)
#define OP_ENABLE    (0x80 | OP_REG)

#define PC_ENABLE      0x01
#define PC_ENABLE_CNA  0x04
#define PC_ENABLE_DPU  0x08

static inline uint64_t npuop(uint32_t op, uint32_t value, uint32_t reg) {
	return ((uint64_t)(op & 0xffff) << 48) | ((uint64_t)value << 16) | (uint64_t)(reg & 0xffff);
}

#define CBUF_BANK_SIZE 32768u
#define CBUF_BANKS     12u

/* precision codes: 0 (int8), 2 (fp16), 4 (int32 out), 5 (fp32 out) from vendor command streams;
 * 3 (bf16) found with tools/prec_probe.c. 1 (int16) hangs the CNA; 5 as an input is not fp32
 * (the CNA reads it as pairs of 16-bit values), so there is no fp32/tf32 MAC path. */
enum { PREC_INT8 = 0, PREC_INT16 = 1, PREC_FP16 = 2, PREC_BF16 = 3, PREC_INT32 = 4, PREC_FP32 = 5 };

#endif
