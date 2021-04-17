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
#include <unistd.h>

#define DEFAULT_SIZE 65536
#define DEFAULT_ITER 10000
#define DEFAULT_WARN 100

char *sbuf, *dbuf;
void *comm_buf, *check_buf /* used for check results */ ;
size_t min_size = 1, max_size = DEFAULT_SIZE, buf_size;
cudaStream_t stream;
cudaIpcMemHandle_t ipc_handle;
int stream_on_dev = 0, map_to_dev = 0;  /* by default use source steram and map to source dev */
int comm_size, comm_rank;

#define CUDA_ERR_ASSERT(cerr) do {              \
    if (cerr != cudaSuccess) {                                                           \
        printf("%s:%d cuda error: %s\n", __func__, __LINE__, cudaGetErrorString(cerr)); \
        assert(cerr == cudaSuccess);                                                    \
    }                                                                                   \
} while (0)

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

static void memcpy_async_stream(size_t off, size_t size)
{
    int cerr = cudaSuccess;
    cerr = cudaMemcpyAsync(dbuf + off, sbuf + off, size, cudaMemcpyDefault, stream);
    CUDA_ERR_ASSERT(cerr);

    cerr = cudaStreamSynchronize(stream);
    CUDA_ERR_ASSERT(cerr);
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

    cerr = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    CUDA_ERR_ASSERT(cerr);

    if (cur_device != device) {
        cerr = cudaSetDevice(cur_device);
        CUDA_ERR_ASSERT(cerr);
    }

    printf("rank %d created stream 0x%lx on device %d\n", comm_rank, stream, device);
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

    /* Default on device rank */
    cerr = cudaSetDevice(comm_rank);
    CUDA_ERR_ASSERT(cerr);

    /* Only rank 0 handles transfer */
    if (comm_rank == 0) {
        create_stream(stream_on_dev == 0 ? 0 : 1);
    }
}

static void cuda_destroy(void)
{
    if (comm_rank == 0) {
        cudaStreamDestroy(stream);
    }
}

static void set_buffer(char *buf, size_t size, char c)
{
    memset(check_buf, c, size);
    cudaMemcpy(buf, check_buf, size, cudaMemcpyDefault);
}

static void reset_buffer(char *buf, size_t size)
{
    memset(check_buf, 'f', size);
    cudaMemcpy(buf, check_buf, size, cudaMemcpyDefault);
}

static void check_buffer(char *buf, size_t size, int niter, size_t stride, char c)
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

static void report_buffer_attr(char *buf)
{
    struct cudaPointerAttributes attr;

    cudaError_t cerr = cudaPointerGetAttributes(&attr, buf);
    CUDA_ERR_ASSERT(cerr);

    printf("rank %d queried attribute of buf=%p: type=%s, device=%d\n",
           comm_rank, buf, attr.type == cudaMemoryTypeDevice ? "GPU" : "other", attr.device);
}

static void exchange_ipc(void **ptr)
{
    cudaError_t cerr;

    /* rank 1 get IPC handle of rbuf */
    /* rank 0 open IPC handle from rank 1 */
    /* rank 0 assign mapped buffer to dbuf */

    if (comm_rank == 1) {
        cerr = cudaIpcGetMemHandle(&ipc_handle, *ptr);
        CUDA_ERR_ASSERT(cerr);
    }

    MPI_Bcast(&ipc_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 1, MPI_COMM_WORLD);

    if (comm_rank == 0) {
        if (map_to_dev == 0) {
            cerr = cudaIpcOpenMemHandle(ptr, ipc_handle, cudaIpcMemLazyEnablePeerAccess);
            CUDA_ERR_ASSERT(cerr);
        } else {
            cerr = cudaSetDevice(1);
            CUDA_ERR_ASSERT(cerr);

            cerr = cudaIpcOpenMemHandle(ptr, ipc_handle, cudaIpcMemLazyEnablePeerAccess);
            CUDA_ERR_ASSERT(cerr);

            cerr = cudaSetDevice(0);
            CUDA_ERR_ASSERT(cerr);
        }
    }
}

static void destroy_ipc(void *ptr)
{
    cudaError_t cerr;
    cerr = cudaIpcCloseMemHandle(ptr);
    CUDA_ERR_ASSERT(cerr);
}

static void init_buffers(void)
{
    cudaError_t cerr;
    buf_size = DEFAULT_ITER * max_size;

    comm_buf = cuda_malloc(buf_size, comm_rank);

    cerr = cudaMallocHost(&check_buf, buf_size);
    CUDA_ERR_ASSERT(cerr);

    if (comm_rank == 0) {
        sbuf = comm_buf;
    } else {
        dbuf = comm_buf;
    }

    printf("rank %d allocated comm_buf=%p, latency for sizes %ld:%ld, buffer size=%ld\n",
           comm_rank, comm_buf, min_size, max_size, buf_size);
    report_buffer_attr(comm_buf);

    /* Rank 0 maps dbuf on rank 1 and assign the mapped addr to local dbuf */
    void *ipc_buf = dbuf;       /* for void casting */
    exchange_ipc(&ipc_buf);
    dbuf = ipc_buf;

    if (comm_rank == 0) {
        printf("rank %d mapped dbuf=%p on device %d\n", comm_rank, dbuf, map_to_dev == 0 ? 0 : 1);
        report_buffer_attr(dbuf);
    }
}

static void free_buffers(void)
{
    if (comm_rank == 0)
        destroy_ipc(dbuf);
    cuda_free(comm_buf);
    cudaFreeHost(check_buf);
}

static void set_params(int argc, char **argv)
{
    int c;
    char *mins, *maxs;
    while ((c = getopt(argc, argv, "s:t:m:")) != -1)
        switch (c) {
            case 's':
                mins = strtok(optarg, ":");
                maxs = strtok(NULL, ":");
                if (mins && maxs) {
                    min_size = atoll(mins);
                    max_size = atoll(maxs);
                }
                break;
            case 't':
                stream_on_dev = atoi(optarg);
                break;
            case 'm':
                map_to_dev = atoi(optarg);
                break;
            default:
                printf("Unknown option %c\n", optopt);
                abort();
                break;
        }
}

int main(int argc, char **argv)
{
    set_params(argc, argv);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    if (min_size < 1 || max_size < min_size) {
        printf("wrong min_size %ld or wrong max_size %ld\n", min_size, max_size);
        goto exit;
    }

    if (comm_size < 2) {
        printf("require at least two processes, comm_size=%d\n", comm_size);
        goto exit;
    }

    cuda_init();
    init_buffers();

    for (int size = min_size; size <= max_size; size *= 2) {

        MPI_Barrier(MPI_COMM_WORLD);    /* ensure buffers are ready to update */

        /* reset buffer for all iterations */
        if (comm_rank == 0) {
            set_buffer(sbuf, buf_size, 'a');
        } else {
            reset_buffer(dbuf, buf_size);
        }

        MPI_Barrier(MPI_COMM_WORLD);    /* ensure buffers have been updated */

        /* rank 0 performs data transfer similar to RMA */
        if (comm_rank == 0) {
            for (int iter = 0; iter < DEFAULT_WARN; iter++) {
#if defined(TEST_MEMCPY_ASYNC)
                memcpy_async(max_size * iter, size);
#elif defined(TEST_MEMCPY_ASYNC_STREAM)
                memcpy_async_stream(max_size * iter, size);
#else
                memcpy_sync(max_size * iter, size);
#endif
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);    /* ensure data transfer is done on rank 0 */
        if (comm_rank == 1) {
            /* touch and check destination buffer */
            check_buffer(dbuf, size, DEFAULT_WARN, max_size, 'a');
        }

        MPI_Barrier(MPI_COMM_WORLD);    /* ensure buffers are ready to update */

        /* reset buffer for all iterations */
        if (comm_rank == 0) {
            set_buffer(sbuf, buf_size, size);
        } else {
            reset_buffer(dbuf, buf_size);
        }

        MPI_Barrier(MPI_COMM_WORLD);    /* ensure buffers have been updated */

        double t0, t1;
        /* rank 0 performs data transfer similar to RMA */
        if (comm_rank == 0) {
            t0 = MPI_Wtime();
            for (int iter = 0; iter < DEFAULT_ITER; iter++) {
#if defined(TEST_MEMCPY_ASYNC)
                memcpy_async(max_size * iter, size);
#elif defined(TEST_MEMCPY_ASYNC_STREAM)
                memcpy_async_stream(max_size * iter, size);
#else /* TEST_MEMCPY_SYNC */
                memcpy_sync(max_size * iter, size);
#endif
            }
            t1 = MPI_Wtime();
        }

        MPI_Barrier(MPI_COMM_WORLD);    /* ensure data transfer is done on rank 0 */

        if (comm_rank == 1) {
            /* touch and check destination buffer */
            check_buffer(dbuf, size, DEFAULT_ITER, max_size, size);
        }

        if (comm_rank == 0) {
            double lat = (t1 - t0) * 1e6 / DEFAULT_ITER;        // in us
            printf("%ld\t %.2f\n", size, lat);
        }
    }

    free_buffers();
    cuda_destroy();

  exit:
    MPI_Finalize();

    return 0;
}
