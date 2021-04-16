/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#define DEFAULT_SIZE 65536
#define DEFAULT_ITER 10000
#define DEFAULT_WARN 100

char *sbuf, *dbuf;
void *tmpbuf;
size_t min_size = 1, max_size = DEFAULT_SIZE, buf_size;
cudaStream_t streams[2];

#define CUDA_ERR_ASSERT(cerr) assert(cerr == cudaSuccess)

static void memcpy_sync(size_t off, size_t size)
{
    int cerr = cudaSuccess;
    cerr = cudaMemcpy(dbuf + off, sbuf + off, size, cudaMemcpyDefault);
    CUDA_ERR_ASSERT(cerr);
}

static void memcpy_async(size_t off, size_t size)
{
    int cerr = cudaSuccess;
    cerr = cudaMemcpyAsync(dbuf + off, sbuf + off, size, cudaMemcpyDefault, 0);
    CUDA_ERR_ASSERT(cerr);

    cerr = cudaStreamSynchronize(0);
    CUDA_ERR_ASSERT(cerr);
}

static void memcpy_async_stream(size_t off, size_t size, int device)
{
    int cerr = cudaSuccess;
    cerr = cudaMemcpyAsync(dbuf + off, sbuf + off, size, cudaMemcpyDefault, streams[device]);
    CUDA_ERR_ASSERT(cerr);

    cerr = cudaStreamSynchronize(streams[device]);
    CUDA_ERR_ASSERT(cerr);
}

static void enable_p2p(int device)
{
    int access = 0;
    cudaError_t cerr;

    int cur_device;
    cerr = cudaGetDevice(&cur_device);
    CUDA_ERR_ASSERT(cerr);

    if (device != cur_device) {
        cerr = cudaDeviceCanAccessPeer(&access, cur_device, device);
        CUDA_ERR_ASSERT(cerr);

        if (access) {
            cerr = cudaDeviceEnablePeerAccess(device, 0);
            CUDA_ERR_ASSERT(cerr);
        }
    }
}

static void *cuda_malloc(size_t size, int device)
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

static void cuda_free(void *ptr)
{
    cudaError_t cerr = cudaFree(ptr);
    CUDA_ERR_ASSERT(cerr);
}

static void create_stream(int device)
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

    cerr = cudaStreamCreateWithFlags(&streams[device], cudaStreamNonBlocking);
    CUDA_ERR_ASSERT(cerr);

    if (cur_device != device) {
        cerr = cudaSetDevice(cur_device);
        CUDA_ERR_ASSERT(cerr);
    }
}

static void cuda_init(void)
{
    cudaError_t cerr;
    int ndevices;

    cerr = cudaGetDeviceCount(&ndevices);
    CUDA_ERR_ASSERT(cerr);

    if (ndevices < 2) {
        printf("This test requires at least 2 devices, found %d\n", ndevices);
        assert(ndevices >= 2);
    }

    /* Default on device 0 */
    cerr = cudaSetDevice(0);
    CUDA_ERR_ASSERT(cerr);

    enable_p2p(1);
    create_stream(0);
    create_stream(1);
}

static void cuda_destroy(void)
{
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
}

static void set_buffer(char *buf, size_t size, char c)
{
    memset(tmpbuf, c, size);
    cudaMemcpy(buf, tmpbuf, size, cudaMemcpyDefault);
}

static void reset_buffer(char *buf, size_t size)
{
    memset(tmpbuf, 'f', size);
    cudaMemcpy(buf, tmpbuf, size, cudaMemcpyDefault);
}

static void check_buffer(char *buf, size_t size, int niter, size_t stride, char c)
{
    char *tmpbuf_c = (char *) tmpbuf;

    cudaMemcpy(tmpbuf_c, buf, niter * stride, cudaMemcpyDefault);
    for (int iter; iter < niter; iter++) {
        for (int i; i < size; i++) {
            if (tmpbuf_c[iter * stride + i] != c) {
                printf("expected %c at buf(%p)[%d= iter %d * stride %ld + %ld], but received %c\n",
                       c, buf, iter * stride + i, iter, stride, i, tmpbuf_c[iter * stride + i]);
                assert(tmpbuf_c[iter * stride + i] == c);
            }
        }
    }
}

static void report_buffer_attr(char *buf)
{
    struct cudaPointerAttributes attr;

    cudaError_t cerr = cudaPointerGetAttributes(&attr, buf);
    CUDA_ERR_ASSERT(cerr);

    printf("queried attribute of buf=%p: type=%s, device=%d\n",
           buf, attr.type == cudaMemoryTypeDevice ? "GPU" : "other", attr.device);
}

static void init_buffers(void)
{
    cudaError_t cerr;
    buf_size = DEFAULT_ITER * max_size;

    sbuf = cuda_malloc(buf_size, 0);
    dbuf = cuda_malloc(buf_size, 1);

    cerr = cudaMallocHost(&tmpbuf, buf_size);
    CUDA_ERR_ASSERT(cerr);

    printf("sbuf=%p, dbuf=%p, latency for sizes %ld:%ld, buffer size=%ld\n",
           sbuf, dbuf, min_size, max_size, buf_size);
    report_buffer_attr(sbuf);
    report_buffer_attr(dbuf);
}

static void free_buffers(void)
{
    cuda_free(sbuf);
    cuda_free(dbuf);
    cudaFreeHost(tmpbuf);
}

int main(int argc, char **argv)
{
    if (argc > 2) {
        min_size = atoi(argv[1]);
        max_size = atoi(argv[2]);
    }

    if (min_size < 1 || max_size < min_size) {
        printf("wrong min_size %ld or wrong max_size %ld\n", min_size, max_size);
        return -1;
    }

    cuda_init();
    init_buffers();

    for (int size = min_size; size <= max_size; size *= 2) {

        /* reset buffer for all iterations */
        set_buffer(sbuf, buf_size, 'a');
        reset_buffer(dbuf, buf_size);
        for (int iter = 0; iter < DEFAULT_WARN; iter++) {
#if defined(TEST_MEMCPY_ASYNC)
            memcpy_async(max_size * iter, size);
#elif defined(TEST_MEMCPY_ASYNC_STREAM_SRC)
            memcpy_async_stream(max_size * iter, size, 0);
#elif defined(TEST_MEMCPY_ASYNC_STREAM_DST)
            memcpy_async_stream(max_size * iter, size, 1);
#else
            memcpy_sync(max_size * iter, size);
#endif
        }
        /* touch and check destination buffer */
        check_buffer(dbuf, size, DEFAULT_WARN, max_size, 'a');

        /* reset buffer for all iterations */
        set_buffer(sbuf, buf_size, size);
        reset_buffer(dbuf, buf_size);

        double t0 = MPI_Wtime();
        for (int iter = 0; iter < DEFAULT_ITER; iter++) {
#if defined(TEST_MEMCPY_ASYNC)
            memcpy_async(max_size * iter, size);
#elif defined(TEST_MEMCPY_ASYNC_STREAM_SRC)
            memcpy_async_stream(max_size * iter, size, 0);
#elif defined(TEST_MEMCPY_ASYNC_STREAM_DST)
            memcpy_async_stream(max_size * iter, size, 1);
#else /* TEST_MEMCPY_SYNC */
            memcpy_sync(max_size * iter, size);
#endif
        }
        double t1 = MPI_Wtime();

        /* touch and check destination buffer */
        check_buffer(dbuf, size, DEFAULT_ITER, max_size, size);

        double lat = (t1 - t0) * 1e6 / DEFAULT_ITER;    // in us
        printf("%ld\t %.2f\n", size, lat);
    }

    free_buffers();
    cuda_destroy();

    return 0;
}
