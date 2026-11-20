CC ?= gcc
MPICC ?= mpicc
CFLAGS ?= -O3 -std=c11 -fopenmp -Wall -Wextra
LDFLAGS ?= -lm

.PHONY: all clean

all: sequential hybrid

sequential: src/sequential.c src/common.c src/common.h
	$(CC) $(CFLAGS) -o $@ src/sequential.c src/common.c $(LDFLAGS)

hybrid: src/hybrid.c src/common.c src/common.h
	$(MPICC) $(CFLAGS) -o $@ src/hybrid.c src/common.c $(LDFLAGS)

clean:
	rm -f sequential hybrid
