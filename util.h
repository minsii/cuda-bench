/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>
#include <mpi.h>

extern char TESTNAME[128];

#define CUDA_ERR_ASSERT(cerr) do {              \
    if (cerr != cudaSuccess) {                                                           \
        printf("%s:%d cuda error: %s\n", __func__, __LINE__, cudaGetErrorString(cerr)); \
        assert(cerr == cudaSuccess);                                                    \
    }                                                                                   \
} while (0)

void get_testname(void);
int enable_p2p(int device, int remote_device);
void disable_p2p(int device, int remote_device);
void *cuda_malloc(size_t size, int device);
void cuda_free(void *ptr);
cudaStream_t create_stream(int device);
void destroy_stream(int device, cudaStream_t stream);

void set_buffer(char *buf, size_t size, char c, void *check_buf);
void reset_buffer(char *buf, size_t size, void *check_buf);
void check_buffer(char *buf, size_t size, int niter, size_t stride, char c, void *check_buf);
void report_buffer_attr(char *buf);

#endif /* UTIL_H */
