OBJS = cudamemcpy_latency_sync cudamemcpy_latency_async cudamemcpy_latency_async_stream	\
			 yaksacopy_latency_pack yaksacopy_latency_unpack \
			 ipc_latency_sync ipc_latency_async ipc_latency_async_stream

SRC = cudamemcpy_latency.c yaksacopy_latency.c ipc_latency.c
CC ?= mpicc
YAKSA_PATH ?= .
CFLAGS = -O2 -g -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcuda -lcudart
YAKSA_CFLAGS=$(CFLAGS) -I$(YAKSA_PATH)/include
YAKSA_LDFLAGS=$(LDFLAGS) -L$(YAKSA_PATH)/lib -Wl,-rpath -Wl,$(YAKSA_PATH)/lib -lyaksa

yaksacopy_latency_pack: yaksacopy_latency.c
	$(CC) -o $@ $< $(YAKSA_CFLAGS) -DTEST_PACK $(YAKSA_LDFLAGS)

yaksacopy_latency_unpack: yaksacopy_latency.c
	$(CC) -o $@ $< $(YAKSA_CFLAGS) $(YAKSA_LDFLAGS)

cudamemcpy_latency_sync: cudamemcpy_latency.c
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)

cudamemcpy_latency_async: cudamemcpy_latency.c
	$(CC) -o $@ $< $(CFLAGS) -DTEST_MEMCPY_ASYNC $(LDFLAGS)

cudamemcpy_latency_async_stream: cudamemcpy_latency.c
	$(CC) -o $@ $< $(CFLAGS) -DTEST_MEMCPY_ASYNC_STREAM $(LDFLAGS)

ipc_latency_sync: ipc_latency.c
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)

ipc_latency_async: ipc_latency.c
	$(CC) -o $@ $< $(CFLAGS) -DTEST_MEMCPY_ASYNC $(LDFLAGS)

ipc_latency_async_stream: ipc_latency.c
	$(CC) -o $@ $< $(CFLAGS) -DTEST_MEMCPY_ASYNC_STREAM $(LDFLAGS)

all: $(OBJS)

clean:
	rm -f $(OBJS)
