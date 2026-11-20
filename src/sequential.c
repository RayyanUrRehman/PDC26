#include "common.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s --size N [--seed S]\n"
            "Single-thread baseline. OpenMP is compiled in and pinned to 1 thread.\n",
            argv0);
}

static int take_int(int argc, char **argv, int *i, long long *out) {
    if (*i + 1 >= argc) {
        return -1;
    }
    char *end = NULL;
    long long v = strtoll(argv[*i + 1], &end, 10);
    if (!end || *end != '\0') {
        return -1;
    }
    *out = v;
    *i += 1;
    return 0;
}

int main(int argc, char **argv) {
    long long N = 0;
    unsigned long long seed = PROJECT_SEED_DEFAULT;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--size") == 0) {
            if (take_int(argc, argv, &i, &N) != 0 || N <= 0 || N > 100000) {
                fprintf(stderr, "invalid --size\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--seed") == 0) {
            long long s = 0;
            if (take_int(argc, argv, &i, &s) != 0 || s < 0) {
                fprintf(stderr, "invalid --seed\n");
                return 1;
            }
            seed = (unsigned long long)s;
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }
    if (N <= 0) {
        fprintf(stderr, "--size is required\n");
        usage(argv[0]);
        return 1;
    }

    long long bytes = 3LL * N * N * (long long)sizeof(double);
    long long avail = read_mem_available_bytes();
    printf("memory_estimate_mib=%.1f available_mib=%.1f\n",
           (double)bytes / (1024.0 * 1024.0),
           avail > 0 ? (double)avail / (1024.0 * 1024.0) : -1.0);
    if (avail > 0 && bytes > avail * 3 / 4) {
        fprintf(stderr,
                "refusing N=%lld: three matrices need %.1f MiB and only %.1f MiB is available\n",
                N, (double)bytes / (1024.0 * 1024.0),
                (double)avail / (1024.0 * 1024.0));
        return 2;
    }

    size_t nn = (size_t)N * (size_t)N;
    double *A = alloc_doubles(nn);
    double *B = alloc_doubles(nn);
    double *C = alloc_doubles(nn);
    if (!A || !B || !C) {
        free(A);
        free(B);
        free(C);
        return 1;
    }

    unsigned long long state = seed;
    lcg_fill(A, nn, &state);
    lcg_fill(B, nn, &state);

    int procs = omp_get_num_procs();
    int offered = omp_get_max_threads();
    printf("cores=1 processes=1 threads=1 procs_available=%d max_threads_offered=%d\n",
           procs, offered);

    double t0 = omp_get_wtime();
    MatmulInfo info;
    matmul_block(A, B, C, (int)N, (int)N, 1, SCHED_STATIC, 0, 0, &info);
    double t_compute = omp_get_wtime() - t0;
    double gflops = (t_compute > 0.0)
                        ? ((2.0 * (double)N) * (double)N * (double)N) / t_compute / 1e9
                        : 0.0;
    double checksum = strided_checksum(C, 0, (int)N, (int)N);

    printf("team_seen=%d last_row=%d clock=omp_get_wtime\n", info.team_seen, info.last_row);
    printf("t_scatter=0.000000 t_bcast=0.000000 t_compute=%.6f t_gather=0.000000 t_total=%.6f\n",
           t_compute, t_compute);
    printf("gflops=%.6f checksum=%.6e max_abs_err=0.000000e+00\n", gflops, checksum);
    printf("csv,%lld,1,1,0.000000,0.000000,%.6f,0.000000,%.6f,%.6f,0.000000e+00,%.6e,static,sequential,0\n",
           N, t_compute, t_compute, gflops, checksum);
    fflush(stdout);

    free(A);
    free(B);
    free(C);
    return info.team_seen == 1 ? 0 : 1;
}
