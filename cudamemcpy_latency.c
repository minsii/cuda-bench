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

#define DEFAULT_SIZE 65536
#define DEFAULT_ITER 10000
#define DEFAULT_WARN 100

char *sbuf, *dbuf;
size_t min_size = 1;
size_t max_size = DEFAULT_SIZE;
cudaStream_t streams[2];

#define CUDA_ERR_ASSERT(cerr) assert(cerr == cudaSuccess)

static void memcpy_sync(size_t size)
{
    int cerr = cudaSuccess;
    cerr = cudaMemcpy(dbuf, sbuf, size, cudaMemcpyDefault);
    CUDA_ERR_ASSERT(cerr);
}

static void memcpy_async(size_t size)
{
    int cerr = cudaSuccess;
    cerr = cudaMemcpyAsync(dbuf, sbuf, size, cudaMemcpyDefault, 0);
    CUDA_ERR_ASSERT(cerr);

    cerr = cudaStreamSynchronize(0);
    CUDA_ERR_ASSERT(cerr);
}

static void memcpy_async_stream(size_t size, int device)
{
    int cerr = cudaSuccess;
    cerr = cudaMemcpyAsync(dbuf, sbuf, size, cudaMemcpyDefault, streams[device]);
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

    cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking);
}

static void create_stream(int device, cudaStream_t *stream)
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

    cerr = cudaStreamCreateWithFlags(&stream[device], cudaStreamNonBlocking)
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

    if(ndevices < 2) {
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

    sbuf = cuda_malloc(max_size, 0);
    dbuf = cuda_malloc(max_size, 1);

    for (int size = min_size; size <= max_size; size *= 2) {

        for (int iter = 0; iter <= DEFAULT_WARN; iter++) {
#if defined(TEST_MEMCPY_ASYNC)
            memcpy_async(size);
#elif defined(TEST_MEMCPY_ASYNC_STREAM_SRC)
            memcpy_async_stream(size, 0);
#elif defined(TEST_MEMCPY_ASYNC_STREAM_DST)
            memcpy_async_stream(size, 1);
#else
            memcpy_sync(size);
#endif
        }

        double t0 = MPI_Wtime();
        for (int iter = 0; iter <= DEFAULT_ITER; iter++) {
#if defined(TEST_MEMCPY_ASYNC)
            memcpy_async(size);
#elif defined(TEST_MEMCPY_ASYNC_STREAM_SRC)
            memcpy_async_stream(size, 0);
#elif defined(TEST_MEMCPY_ASYNC_STREAM_DST)
            memcpy_async_stream(size, 1);
#else /* TEST_MEMCPY_SYNC */
            memcpy_sync(size);
#endif
        }
        double t1 = MPI_Wtime();

        double lat = (t1 - t0) * 1e6 / DEFAULT_ITER;    // in us
        printf("%ld\t %.2f\n", size, lat);
    }

    cuda_free(sbuf);
    cuda_free(dbuf);

    cuda_destroy();

    return 0;
}
