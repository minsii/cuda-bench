OBJS = cudamemcpy_latency_sync cudamemcpy_latency_async cudamemcpy_latency_async_stream_src cudamemcpy_latency_async_stream_dst
SRC = cudamemcpy_latency.c
CC = mpicc
CFLAGS = -O2 -g -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64

cudamemcpy_latency_sync: $(SRC)
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)

cudamemcpy_latency_async: $(SRC)
	$(CC) -o $@ $< $(CFLAGS) -DTEST_MEMCPY_ASYNC $(LDFLAGS)

cudamemcpy_latency_async_stream_src: $(SRC)
	$(CC) -o $@ $< $(CFLAGS) -DTEST_MEMCPY_ASYNC_STREAM_SRC $(LDFLAGS)

cudamemcpy_latency_async_stream_dst: $(SRC)
	$(CC) -o $@ $< $(CFLAGS) -DTEST_MEMCPY_ASYNC_STREAM_DST $(LDFLAGS)

all: $(OBJS)

clean:
	rm -f $(OBJS)
