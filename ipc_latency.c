/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include "util.h"

#define DEFAULT_SIZE 65536
#define DEFAULT_ITER 10000
#define DEFAULT_WARM 100

char *sbuf, *dbuf;
void *comm_buf, *ipc_buf, *check_buf; /* used for check results */ ;
size_t min_size = 1, max_size = DEFAULT_SIZE, buf_size;
cudaStream_t stream;
cudaIpcMemHandle_t ipc_handle;
int stream_on_dev = 0, map_to_dev = 0;  /* by default use source steram and map to source dev */
int comm_size, comm_rank;
int nwarm = DEFAULT_WARM, niter = DEFAULT_ITER;
int s_devid = 0, d_devid = 1;
int enabled_p2p[2];             /* flag to mark whether this test enabled P2P for dev0->dev1 and
                                 * dev1->dev0 direction. Used for disable P2P */

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

    nvtxNameCudaDeviceA(0, "dev0");
    nvtxNameCudaDeviceA(1, "dev1");

    if (comm_rank == 0) {
        /* Enable P2P for both dev0->dev1 and dev1->dev0 */
        enabled_p2p[0] = enable_p2p(0, 1);
        enabled_p2p[1] = enable_p2p(1, 0);
    }
#ifdef TEST_MEMCPY_ASYNC_STREAM
    /* Only rank 0 handles transfer */
    if (comm_rank == 0) {
        stream = create_stream(stream_on_dev);

        char streamname[100];
        sprintf(streamname, "dev%d-stream", stream_on_dev);
        nvtxNameCudaStreamA(stream, streamname);
        printf("rank 0 created %s\n", streamname);
    }
#endif
}

static void cuda_destroy(void)
{
#ifdef TEST_MEMCPY_ASYNC_STREAM
    if (comm_rank == 0) {
        destroy_stream(stream_on_dev, stream);
    }
#endif
    if (comm_rank == 0) {
        if (enabled_p2p[0])
            disable_p2p(0, 1);
        if (enabled_p2p[1])
            disable_p2p(1, 0);
    }
}

static void exchange_ipc(void **ptr)
{
    cudaError_t cerr;

    /* rank 1 get IPC handle of rbuf */
    if (comm_rank == 1) {
        cerr = cudaIpcGetMemHandle(&ipc_handle, *ptr);
        CUDA_ERR_ASSERT(cerr);
    }

    MPI_Bcast(&ipc_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 1, MPI_COMM_WORLD);

    /* rank 0 open IPC handle received from rank 1 */
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
    buf_size = niter * max_size;

    comm_buf = cuda_malloc(buf_size, comm_rank);

    cerr = cudaMallocHost(&check_buf, buf_size);
    CUDA_ERR_ASSERT(cerr);

    printf("rank %d allocated comm_buf=%p, latency for sizes %ld:%ld, buffer size=%ld\n",
           comm_rank, comm_buf, min_size, max_size, buf_size);
    report_buffer_attr(comm_buf);

    /* Rank 0 maps comm_buf on rank 1 and assign the mapped addr to local ipc_buf */
    ipc_buf = comm_buf; /* for avoid casting */
    exchange_ipc(&ipc_buf);

    if (comm_rank == 0) {
        sbuf = s_devid == 0 ? comm_buf : ipc_buf;
        dbuf = d_devid == 0 ? comm_buf : ipc_buf;

        printf("rank %d mapped ipc_buf=%p on device %d\n", comm_rank, ipc_buf, map_to_dev);

        printf("rank %d set sbuf %p on dev %d\n", comm_rank, sbuf, s_devid);
        report_buffer_attr(sbuf);
        printf("rank %d set dbuf %p on dev %d\n", comm_rank, dbuf, d_devid);
        report_buffer_attr(dbuf);
    }
}

static void free_buffers(void)
{
    if (comm_rank == 0)
        destroy_ipc(ipc_buf);
    cuda_free(comm_buf);
    cudaFreeHost(check_buf);
}

static void set_params(int argc, char **argv)
{
    int c;
    char *mins, *maxs, *iter, *warm, *src_devid, *dst_devid;
    while ((c = getopt(argc, argv, "s:t:m:i:d:h")) != -1) {
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
                stream_on_dev = stream_on_dev == 0 ? 0 : 1;
                break;
            case 'm':
                map_to_dev = atoi(optarg);
                map_to_dev = map_to_dev == 0 ? 0 : 1;
                break;
            case 'i':
                warm = strtok(optarg, ":");
                iter = strtok(NULL, ":");
                if (warm && iter) {
                    nwarm = atoi(warm);
                    niter = atoi(iter);
                }
                break;
            case 'd':
                src_devid = strtok(optarg, ":");
                dst_devid = strtok(NULL, ":");
                if (src_devid && dst_devid) {
                    s_devid = atoi(src_devid);
                    d_devid = atoi(dst_devid);
                }
                break;
            case 'h':
                printf("./ipc_latency -s <message size, format min:max> \\\n"
                       "    -t <stream on device, value 0|1>\\\n"
                       "    -m <map to device, value 0|1>\\\n"
                       "    -i <number of warmming up: number of iteration>\\\n"
                       "    -d <src device:dst device|default 0:1>\n");
                fflush(stdout);
                abort();
                break;
            default:
                printf("Unknown option %c\n", optopt);
                fflush(stdout);
                abort();
                break;
        }
    }

    if (min_size < 1 || max_size < min_size) {
        printf("wrong min_size %ld or wrong max_size %ld\n", min_size, max_size);
        abort();
    }
}

int main(int argc, char **argv)
{
    set_params(argc, argv);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    if (comm_size < 2) {
        printf("require at least two processes, comm_size=%d\n", comm_size);
        goto exit;
    }

    get_testname();
    cuda_init();
    init_buffers();

    MPI_Barrier(MPI_COMM_WORLD);

    /* let rank 0 perform all data transfer and data setup because
     * it dynamically setups sbuf and dbuf */
    if (comm_rank == 0) {
        for (int size = min_size; size <= max_size; size *= 2) {
            /* reset buffer for all iterations */
            set_buffer(sbuf, buf_size, 'a', check_buf);
            reset_buffer(dbuf, buf_size, check_buf);

            for (int iter = 0; iter < nwarm; iter++) {
#if defined(TEST_MEMCPY_ASYNC)
                memcpy_async(max_size * iter, size);
#elif defined(TEST_MEMCPY_ASYNC_STREAM)
                memcpy_async_stream(max_size * iter, size);
#else
                memcpy_sync(max_size * iter, size);
#endif
            }

            /* touch and check destination buffer */
            check_buffer(dbuf, size, nwarm, max_size, 'a', check_buf);

            /* reset buffer for all iterations */
            set_buffer(sbuf, buf_size, size, check_buf);
            reset_buffer(dbuf, buf_size, check_buf);

            double t0, t1;

            cudaProfilerStart();
            t0 = MPI_Wtime();
            for (int iter = 0; iter < niter; iter++) {
#if defined(TEST_MEMCPY_ASYNC)
                memcpy_async(max_size * iter, size);
#elif defined(TEST_MEMCPY_ASYNC_STREAM)
                memcpy_async_stream(max_size * iter, size);
#else /* TEST_MEMCPY_SYNC */
                memcpy_sync(max_size * iter, size);
#endif
            }
            t1 = MPI_Wtime();
            cudaProfilerStop();

            /* touch and check destination buffer */
            check_buffer(dbuf, size, niter, max_size, 'a', check_buf);

            double lat = (t1 - t0) * 1e6 / niter;       // in us
            printf("%s\t%ld\t %.2f\n", TESTNAME, size, lat);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    free_buffers();
    cuda_destroy();

  exit:
    MPI_Finalize();

    return 0;
}
