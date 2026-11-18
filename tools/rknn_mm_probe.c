/* Run one rknn_matmul of a given type (for capturing command streams with ioctl_spy). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_matmul_api.h"

int main(int argc, char **argv) {
	int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]), type = atoi(argv[4]);
	int pat = argc > 5 ? atoi(argv[5]) : 0;
	rknn_matmul_ctx ctx = 0;
	rknn_matmul_info info; memset(&info, 0, sizeof(info));
	info.M = M; info.K = K; info.N = N; info.type = (rknn_matmul_type)type;
	rknn_matmul_io_attr io; memset(&io, 0, sizeof(io));
	int r = rknn_matmul_create(&ctx, &info, &io);
	if (r) { printf("type %d: create failed %d\n", type, r); return 1; }
	printf("type %d (%s): A %u B %u C %u bytes, A type %d B type %d C type %d\n", type, get_matmul_type_string(info.type),
	       io.A.size, io.B.size, io.C.size, io.A.type, io.B.type, io.C.type);
	rknn_tensor_mem *a = rknn_create_mem(ctx, io.A.size), *b = rknn_create_mem(ctx, io.B.size), *c = rknn_create_mem(ctx, io.C.size);
	unsigned char *pa = a->virt_addr, *pb = b->virt_addr;
	srand(1);
	int a_fp16 = io.A.type == RKNN_TENSOR_FLOAT16, b_fp16 = io.B.type == RKNN_TENSOR_FLOAT16;
	if (a_fp16) for (unsigned i = 0; i < io.A.size / 2; i++) ((_Float16 *)pa)[i] = (_Float16)((rand() % 9) - 4);
	else for (unsigned i = 0; i < io.A.size; i++) pa[i] = (unsigned char)((rand() % 9) - 4);
	if (pat == 1) {           /* B[k][n] = k  (int8 / int4 low nibble)   */
		for (int k = 0; k < K; k++) for (int n = 0; n < N; n++) {
			if (b_fp16) ((_Float16 *)pb)[k * N + n] = (_Float16)k; else if (io.B.type == RKNN_TENSOR_INT8) ((signed char *)pb)[k * N + n] = (signed char)(k & 127);
		}
	} else if (pat == 2) {    /* B[k][n] = n */
		for (int k = 0; k < K; k++) for (int n = 0; n < N; n++) {
			if (b_fp16) ((_Float16 *)pb)[k * N + n] = (_Float16)n; else if (io.B.type == RKNN_TENSOR_INT8) ((signed char *)pb)[k * N + n] = (signed char)(n & 127);
		}
	} else if (b_fp16) for (unsigned i = 0; i < io.B.size / 2; i++) ((_Float16 *)pb)[i] = (_Float16)((rand() % 9) - 4);
	else for (unsigned i = 0; i < io.B.size; i++) pb[i] = (unsigned char)(((rand() % 9) - 4) & (io.B.type == RKNN_TENSOR_INT4 ? 0x77 : 0xff));
	rknn_matmul_set_io_mem(ctx, a, &io.A); rknn_matmul_set_io_mem(ctx, b, &io.B); rknn_matmul_set_io_mem(ctx, c, &io.C);
	r = rknn_matmul_run(ctx);
	printf("type %d run %d\n", type, r);
	rknn_destroy_mem(ctx, a); rknn_destroy_mem(ctx, b); rknn_destroy_mem(ctx, c);
	rknn_matmul_destroy(ctx);
	return 0;
}
