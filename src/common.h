#ifndef COMMON_H
#define COMMON_H

#include <stddef.h>

#define CHECKSUM_STRIDE 1024
#define VERIFY_ABS_TOL 1e-6
#define FULL_VERIFY_MAX_N 512
#define PARALLEL_ROW_THRESHOLD 32
#define PROJECT_SEED_DEFAULT 42ULL

enum {
    SCHED_STATIC = 0,
    SCHED_STATIC_CHUNK = 1,
    SCHED_DYNAMIC = 2,
    SCHED_GUIDED = 3,
    SCHED_RUNTIME = 4
};

typedef struct {
    int team_seen;
    int last_row;
} MatmulInfo;

double *alloc_doubles(size_t count);
void lcg_fill(double *dest, size_t count, unsigned long long *state);

void matmul_block(const double *A, const double *B, double *C,
                  int rows, int N, int threads, int schedule, int chunk,
                  int trace, MatmulInfo *info);

double strided_checksum(const double *C, int row_start, int rows, int N);
double max_abs_diff(const double *x, const double *y, size_t count);
long long read_mem_available_bytes(void);

int parse_schedule(const char *text, int *schedule, int *chunk);
const char *schedule_name(int schedule, int chunk);

#endif
