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
void *check_buf;
size_t min_size = 1, max_size = DEFAULT_SIZE, buf_size;
cudaStream_t stream;
int stream_on_dev = 0;
int niter = DEFAULT_ITER;
int nwarm = DEFAULT_WARM;
int s_devid = 0, d_devid = 1;

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

    /* Default on device 0 */
    cerr = cudaSetDevice(0);
    CUDA_ERR_ASSERT(cerr);

    nvtxNameCudaDeviceA(0, "dev0");
    nvtxNameCudaDeviceA(1, "dev1");

    /* enable P2P for dev0->dev1 and dev1->dev0 */
    enable_p2p(0, 1);
    enable_p2p(1, 0);

#ifdef TEST_MEMCPY_ASYNC_STREAM
    stream = create_stream(stream_on_dev);

    char streamname[100];
    sprintf(streamname, "dev%d-stream", stream_on_dev);
    nvtxNameCudaStreamA(stream, streamname);
    printf("rank 0 created %s\n", streamname);
#endif
}

static void cuda_destroy(void)
{
#ifdef TEST_MEMCPY_ASYNC_STREAM
    cudaStreamDestroy(stream);
#endif
    disable_p2p(0, 1);
    disable_p2p(1, 0);
}

static void init_buffers(void)
{
    cudaError_t cerr;
    buf_size = niter * max_size;
    sbuf = cuda_malloc(buf_size, s_devid);
    dbuf = cuda_malloc(buf_size, d_devid);
    cerr = cudaMallocHost(&check_buf, buf_size);
    CUDA_ERR_ASSERT(cerr);
    printf("sbuf=%p from dev %d, dbuf=%p from dev %d,"
           "latency for sizes %ld:%ld, buffer size=%ld, nwarm=%d, niter=%d\n",
           sbuf, s_devid, dbuf, d_devid, min_size, max_size, buf_size, nwarm, niter);
    report_buffer_attr(sbuf);
    report_buffer_attr(dbuf);
}

static void free_buffers(void)
{
    cuda_free(sbuf);
    cuda_free(dbuf);
    cudaFreeHost(check_buf);
}

static void set_params(int argc, char **argv)
{
    int c;
    char *mins, *maxs, *iter, *warm, *src_devid, *dst_devid;
    while ((c = getopt(argc, argv, "s:t:i:d:h")) != -1) {
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
                printf("./cudamemcpy_latency -s <message size, format min:max>\\\n"
                       "    -t <stream on device, value 0|1>\\\n"
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
    get_testname();
    cuda_init();
    init_buffers();
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
        cudaProfilerStart();
        double t0 = MPI_Wtime();
        for (int iter = 0; iter < niter; iter++) {
#if defined(TEST_MEMCPY_ASYNC)
            memcpy_async(max_size * iter, size);
#elif defined(TEST_MEMCPY_ASYNC_STREAM)
            memcpy_async_stream(max_size * iter, size);
#else /* TEST_MEMCPY_SYNC */
            memcpy_sync(max_size * iter, size);
#endif
        }
        double t1 = MPI_Wtime();
        cudaProfilerStop();
        /* touch and check destination buffer */
        check_buffer(dbuf, size, niter, max_size, size, check_buf);
        double lat = (t1 - t0) * 1e6 / niter;   // in us
        printf("%s\t%ld\t %.2f\n", TESTNAME, size, lat);
    }

    free_buffers();
    cuda_destroy();
    return 0;
}
