/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdlib.h>
#include "util.h"

char TESTNAME[128];

void get_testname(void)
{
    char *str;
    str = getenv("TESTNAME");
    if (str && strlen(str)) {
        memcpy(TESTNAME, str, strlen(str));
    } else {
        memset(TESTNAME, 0, sizeof(TESTNAME));
    }
}

int enable_p2p(int device, int remote_device)
{
    int access = 0, enabled = 0;
    cudaError_t cerr;

    int cur_device;
    cerr = cudaGetDevice(&cur_device);
    CUDA_ERR_ASSERT(cerr);

    cerr = cudaSetDevice(device);
    CUDA_ERR_ASSERT(cerr);

    cerr = cudaDeviceCanAccessPeer(&access, device, remote_device);
    CUDA_ERR_ASSERT(cerr);

    if (access) {
        cerr = cudaDeviceEnablePeerAccess(remote_device, 0);
        if (cerr == cudaErrorPeerAccessAlreadyEnabled) {
            printf("P2P for dev %d->%d is already enabled\n", device, remote_device);
        } else {
            CUDA_ERR_ASSERT(cerr);
            printf("enabled P2P for dev %d->%d\n", device, remote_device);
            enabled = 1;
        }
    }

    cerr = cudaSetDevice(cur_device);
    CUDA_ERR_ASSERT(cerr);
    return enabled;
}

void disable_p2p(int device, int remote_device)
{
    cudaError_t cerr;

    int cur_device;
    cerr = cudaGetDevice(&cur_device);
    CUDA_ERR_ASSERT(cerr);

    cerr = cudaSetDevice(device);
    CUDA_ERR_ASSERT(cerr);

    cerr = cudaDeviceDisablePeerAccess(remote_device);
    CUDA_ERR_ASSERT(cerr);

    cerr = cudaSetDevice(cur_device);
    CUDA_ERR_ASSERT(cerr);
}

void *cuda_malloc(size_t size, int device)
{
    void *ptr = NULL;
    cudaError_t cerr;

    int cur_device;
    cerr = cudaGetDevice(&cur_device);
    CUDA_ERR_ASSERT(cerr);

    if (cur_device != device) {
        cerr = cudaSetDevice(device);
        CUDA_ERR_ASSERT(cerr);
    }

    cerr = cudaMalloc(&ptr, size);
    CUDA_ERR_ASSERT(cerr);

    if (cur_device != device) {
        cerr = cudaSetDevice(cur_device);
        CUDA_ERR_ASSERT(cerr);
    }

    return ptr;
}

void cuda_free(void *ptr)
{
    cudaError_t cerr = cudaFree(ptr);
    CUDA_ERR_ASSERT(cerr);
}

cudaStream_t create_stream(int device)
{
    cudaError_t cerr;
    cudaStream_t stream;

    int cur_device;
    cerr = cudaGetDevice(&cur_device);
    CUDA_ERR_ASSERT(cerr);

    if (cur_device != device) {
        cerr = cudaSetDevice(device);
        CUDA_ERR_ASSERT(cerr);
    }

    cerr = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    CUDA_ERR_ASSERT(cerr);

    if (cur_device != device) {
        cerr = cudaSetDevice(cur_device);
        CUDA_ERR_ASSERT(cerr);
    }

    return stream;
}

void destroy_stream(int device, cudaStream_t stream)
{
    cudaError_t cerr;

    int cur_device;
    cerr = cudaGetDevice(&cur_device);
    CUDA_ERR_ASSERT(cerr);

    if (cur_device != device) {
        cerr = cudaSetDevice(device);
        CUDA_ERR_ASSERT(cerr);
    }

    cerr = cudaStreamDestroy(stream);
    CUDA_ERR_ASSERT(cerr);

    if (cur_device != device) {
        cerr = cudaSetDevice(cur_device);
        CUDA_ERR_ASSERT(cerr);
    }
}

void set_buffer(char *buf, size_t size, char c, void *check_buf)
{
    memset(check_buf, c, size);
    cudaMemcpy(buf, check_buf, size, cudaMemcpyDefault);
}

void reset_buffer(char *buf, size_t size, void *check_buf)
{
    memset(check_buf, 'f', size);
    cudaMemcpy(buf, check_buf, size, cudaMemcpyDefault);
}

void check_buffer(char *buf, size_t size, int niter, size_t stride, char c, void *check_buf)
{
    char *check_buf_c = (char *) check_buf;

    cudaMemcpy(check_buf_c, buf, niter * stride, cudaMemcpyDefault);
    for (int iter; iter < niter; iter++) {
        for (int i; i < size; i++) {
            if (check_buf_c[iter * stride + i] != c) {
                printf("expected %c at buf(%p)[%d= iter %d * stride %ld + %ld], but received %c\n",
                       c, buf, iter * stride + i, iter, stride, i, check_buf_c[iter * stride + i]);
                assert(check_buf_c[iter * stride + i] == c);
            }
        }
    }
}

void report_buffer_attr(char *buf)
{
    struct cudaPointerAttributes attr;

    cudaError_t cerr = cudaPointerGetAttributes(&attr, buf);
    CUDA_ERR_ASSERT(cerr);

    printf("queried attribute of buf=%p: type=%s, device=%d\n",
           buf, attr.type == cudaMemoryTypeDevice ? "GPU" : "other", attr.device);
}
