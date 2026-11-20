#include "common.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double *alloc_doubles(size_t count) {
    double *p = (double *)malloc(count * sizeof(double));
    if (!p) {
        fprintf(stderr, "allocation of %zu doubles failed\n", count);
    }
    return p;
}

void lcg_fill(double *dest, size_t count, unsigned long long *state) {
    unsigned long long x = *state;
    if (x == 0) {
        x = 1ULL;
    }
    for (size_t i = 0; i < count; ++i) {
        x = x * 6364136223846793005ULL + 1ULL;
        double unit = (double)(x >> 11) * (1.0 / 9007199254740992.0);
        dest[i] = unit * 2.0 - 1.0;
    }
    *state = x;
}

long long read_mem_available_bytes(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) {
        return 0;
    }
    char line[256];
    long long kb = 0;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "MemAvailable: %lld kB", &kb) == 1) {
            fclose(f);
            return kb * 1024LL;
        }
    }
    fclose(f);
    return 0;
}

int parse_schedule(const char *text, int *schedule, int *chunk) {
    char buf[64];
    size_t n = strlen(text);
    if (n == 0 || n >= sizeof buf) {
        return -1;
    }
    for (size_t i = 0; i < n; ++i) {
        buf[i] = (char)tolower((unsigned char)text[i]);
    }
    buf[n] = '\0';

    char *comma = strchr(buf, ',');
    int parsed_chunk = 0;
    if (comma) {
        *comma = '\0';
        char *end = NULL;
        long v = strtol(comma + 1, &end, 10);
        if (!end || *end != '\0' || v <= 0 || v > 1000000L) {
            return -1;
        }
        parsed_chunk = (int)v;
    }

    if (strcmp(buf, "static") == 0) {
        if (parsed_chunk > 0) {
            *schedule = SCHED_STATIC_CHUNK;
            *chunk = parsed_chunk;
        } else {
            *schedule = SCHED_STATIC;
            *chunk = 0;
        }
        return 0;
    }
    if (strcmp(buf, "dynamic") == 0) {
        *schedule = SCHED_DYNAMIC;
        *chunk = parsed_chunk > 0 ? parsed_chunk : 1;
        return 0;
    }
    if (strcmp(buf, "guided") == 0) {
        *schedule = SCHED_GUIDED;
        *chunk = parsed_chunk > 0 ? parsed_chunk : 1;
        return 0;
    }
    if (strcmp(buf, "runtime") == 0 && parsed_chunk == 0) {
        *schedule = SCHED_RUNTIME;
        *chunk = 0;
        return 0;
    }
    return -1;
}

const char *schedule_name(int schedule, int chunk) {
    static char buf[32];
    switch (schedule) {
    case SCHED_STATIC:
        return "static";
    case SCHED_STATIC_CHUNK:
        snprintf(buf, sizeof buf, "static_%d", chunk);
        return buf;
    case SCHED_DYNAMIC:
        snprintf(buf, sizeof buf, "dynamic_%d", chunk);
        return buf;
    case SCHED_GUIDED:
        snprintf(buf, sizeof buf, "guided_%d", chunk);
        return buf;
    case SCHED_RUNTIME:
        return "runtime";
    default:
        return "unknown";
    }
}

double strided_checksum(const double *C, int row_start, int rows, int N) {
    double sum = 0.0;
    for (int i = 0; i < rows; ++i) {
        long long base = (long long)(row_start + i) * (long long)N;
        const double *row = C + (size_t)i * (size_t)N;
        for (int j = 0; j < N; ++j) {
            if ((base + j) % CHECKSUM_STRIDE == 0) {
                sum += row[j];
            }
        }
    }
    return sum;
}

double max_abs_diff(const double *x, const double *y, size_t count) {
    double m = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double d = fabs(x[i] - y[i]);
        if (d > m) {
            m = d;
        }
    }
    return m;
}

static void trace_row(int row, int thread) {
    if (row < 4 || row % 1024 == 0) {
        fprintf(stderr, "trace row %d thread %d\n", row, thread);
    }
}

static void row_product(const double *restrict A, const double *restrict B,
                        double *restrict C, int i, int N) {
    for (int k = 0; k < N; ++k) {
        double aik = A[(size_t)i * (size_t)N + (size_t)k];
        double *crow = C + (size_t)i * (size_t)N;
        const double *brow = B + (size_t)k * (size_t)N;
        #pragma omp simd
        for (int j = 0; j < N; ++j) {
            crow[j] += aik * brow[j];
        }
    }
}

void matmul_block(const double *A, const double *B, double *C,
                  int rows, int N, int threads, int schedule, int chunk,
                  int trace, MatmulInfo *info) {
    int team_seen = 1;
    int last_i = -1;
    int n_copy = N;
    int tid = -1;

    if (threads < 1) {
        threads = 1;
    }
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    #pragma omp parallel num_threads(threads) if (rows > PARALLEL_ROW_THRESHOLD) \
        default(none) \
        shared(A, B, C, rows, N, team_seen, last_i, schedule, chunk, trace) \
        firstprivate(n_copy) \
        private(tid)
    {
        tid = omp_get_thread_num();
        (void)tid;

        #pragma omp master
        {
            team_seen = omp_get_num_threads();
        }

        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < N; ++j) {
                C[(size_t)i * (size_t)N + (size_t)j] = 0.0;
            }
        }

        if (trace) {
            #pragma omp for schedule(static) ordered lastprivate(last_i)
            for (int i = 0; i < rows; ++i) {
                row_product(A, B, C, i, n_copy);
                #pragma omp ordered
                trace_row(i, tid);
                last_i = i;
            }
        } else if (schedule == SCHED_STATIC) {
            #pragma omp for schedule(static) lastprivate(last_i)
            for (int i = 0; i < rows; ++i) {
                row_product(A, B, C, i, n_copy);
                last_i = i;
            }
        } else if (schedule == SCHED_STATIC_CHUNK) {
            #pragma omp for schedule(static, chunk) lastprivate(last_i)
            for (int i = 0; i < rows; ++i) {
                row_product(A, B, C, i, n_copy);
                last_i = i;
            }
        } else if (schedule == SCHED_DYNAMIC) {
            #pragma omp for schedule(dynamic, chunk) lastprivate(last_i)
            for (int i = 0; i < rows; ++i) {
                row_product(A, B, C, i, n_copy);
                last_i = i;
            }
        } else if (schedule == SCHED_GUIDED) {
            #pragma omp for schedule(guided, chunk) lastprivate(last_i)
            for (int i = 0; i < rows; ++i) {
                row_product(A, B, C, i, n_copy);
                last_i = i;
            }
        } else {
            #pragma omp for schedule(runtime) lastprivate(last_i)
            for (int i = 0; i < rows; ++i) {
                row_product(A, B, C, i, n_copy);
                last_i = i;
            }
        }
    }

    if (info) {
        info->team_seen = team_seen;
        info->last_row = last_i;
    }
    if (rows > PARALLEL_ROW_THRESHOLD && team_seen != threads) {
        fprintf(stderr,
                "warning: requested %d threads but the team size was %d\n",
                threads, team_seen);
    }
}
