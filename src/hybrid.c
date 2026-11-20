#include "common.h"

#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TAG_ROWS = 11,
    TAG_REPORT = 22,
    TAG_RING = 33,
    REPORT_DOUBLES = 11
};

typedef struct {
    int N;
    unsigned long long seed;
    int threads;
    int schedule;
    int chunk;
    int trace;
    int oversubscribe;
} Options;

typedef struct {
    double crit_sum;
    double red_sum;
    double max_abs;
    double first_row;
    double last_row;
    double packed_row;
    double packed_max;
    double packed_both;
    long long large_count;
    long long huge_count;
    double huge_sum;
    int stats_ok;
} LocalStats;

static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s --size N [--threads T] [--seed S]\n"
            "          [--schedule static|static,C|dynamic[,C]|guided[,C]|runtime]\n"
            "          [--trace] [--oversubscribe 0|1]\n"
            "\n"
            "Process count comes from mpirun. Thread count comes from --threads,\n"
            "otherwise OMP_NUM_THREADS, otherwise 1. It does not default to every core.\n",
            argv0);
}

static int take_ll(int argc, char **argv, int *i, long long *out) {
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

static int parse_options(int argc, char **argv, Options *opt) {
    opt->N = 0;
    opt->seed = PROJECT_SEED_DEFAULT;
    opt->threads = -1;
    opt->schedule = SCHED_STATIC;
    opt->chunk = 0;
    opt->trace = 0;
    opt->oversubscribe = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            return 1;
        } else if (strcmp(argv[i], "--size") == 0) {
            long long v = 0;
            if (take_ll(argc, argv, &i, &v) != 0 || v <= 0 || v > 100000) {
                return -1;
            }
            opt->N = (int)v;
        } else if (strcmp(argv[i], "--threads") == 0) {
            long long v = 0;
            if (take_ll(argc, argv, &i, &v) != 0 || v <= 0 || v > 4096) {
                return -1;
            }
            opt->threads = (int)v;
        } else if (strcmp(argv[i], "--seed") == 0) {
            long long v = 0;
            if (take_ll(argc, argv, &i, &v) != 0 || v < 0) {
                return -1;
            }
            opt->seed = (unsigned long long)v;
        } else if (strcmp(argv[i], "--schedule") == 0) {
            if (i + 1 >= argc || parse_schedule(argv[i + 1], &opt->schedule, &opt->chunk) != 0) {
                return -1;
            }
            i += 1;
        } else if (strcmp(argv[i], "--trace") == 0) {
            opt->trace = 1;
        } else if (strcmp(argv[i], "--oversubscribe") == 0) {
            long long v = 0;
            if (take_ll(argc, argv, &i, &v) != 0 || (v != 0 && v != 1)) {
                return -1;
            }
            opt->oversubscribe = (int)v;
        } else if (strcmp(argv[i], "--verify") == 0) {
            /* Checks always run. Accepted so the documented flag is real. */
        } else {
            return -1;
        }
    }
    if (opt->N <= 0) {
        return -1;
    }
    if (opt->threads < 1) {
        const char *env = getenv("OMP_NUM_THREADS");
        if (env && *env) {
            opt->threads = atoi(env);
        }
        if (opt->threads < 1) {
            opt->threads = 1;
        }
    }
    return 0;
}

static void mpi_check(int rc, const char *what) {
    if (rc == MPI_SUCCESS) {
        return;
    }
    char err[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(rc, err, &len);
    fprintf(stderr, "MPI error in %s: %s\n", what, err);
    MPI_Abort(MPI_COMM_WORLD, rc ? rc : 1);
}

static void row_layout(int N, int P, int rank, int *local_rows, int *row_start) {
    int base = N / P;
    int rem = N % P;
    *local_rows = base + (rank < rem ? 1 : 0);
    *row_start = rank * base + (rank < rem ? rank : rem);
}

static int mpi_elems(int rows, int N, int allow_zero) {
    long long n = (long long)rows * (long long)N;
    if (n < 0 || n > 2147483647LL || (n == 0 && !allow_zero)) {
        fprintf(stderr, "matrix block of %lld elements does not fit in an MPI count\n", n);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    return (int)n;
}

static int sample_count(int N) {
    return N < 8 ? N : 8;
}

static int sample_row_index(int N, int s) {
    int ns = sample_count(N);
    if (N < 8) {
        return s;
    }
    return (int)(((long long)s * (long long)N) / ns);
}

static double row_error(const double *Arow, const double *B, const double *Crow, int N) {
    double err = 0.0;
    for (int j = 0; j < N; ++j) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += Arow[k] * B[(size_t)k * (size_t)N + (size_t)j];
        }
        double d = fabs(sum - Crow[j]);
        if (d > err) {
            err = d;
        }
    }
    return err;
}

static void local_stats(const double *C, int row_start, int rows, int N, int threads,
                        LocalStats *out) {
    double crit_sum = 0.0;
    double red_sum = 0.0;
    double max_abs = 0.0;
    double first_row = 0.0;
    double last_row = 0.0;
    long long large_count = 0;
    long long huge_count = 0;
    double huge_sum = 0.0;
    int stride = CHECKSUM_STRIDE;

    #pragma omp parallel num_threads(threads) default(none) \
        shared(C, row_start, rows, N, crit_sum, first_row, last_row, large_count, huge_count, huge_sum) \
        firstprivate(stride) \
        reduction(+:red_sum) reduction(max:max_abs)
    {
        double local = 0.0;
        #pragma omp for nowait
        for (int i = 0; i < rows; ++i) {
            long long base = (long long)(row_start + i) * (long long)N;
            const double *row = C + (size_t)i * (size_t)N;
            for (int j = 0; j < N; ++j) {
                if ((base + j) % stride == 0) {
                    local += row[j];
                }
            }
        }
        #pragma omp critical(checksum)
        {
            crit_sum += local;
        }

        #pragma omp for
        for (int i = 0; i < rows; ++i) {
            long long base = (long long)(row_start + i) * (long long)N;
            const double *row = C + (size_t)i * (size_t)N;
            for (int j = 0; j < N; ++j) {
                double v = row[j];
                double av = fabs(v);
                if (av > max_abs) {
                    max_abs = av;
                }
                if ((base + j) % stride == 0) {
                    red_sum += v;
                }
                if (av > 1.0 && ((base + j) % 64) == 0) {
                    #pragma omp atomic update
                    large_count += 1;
                }
                if (av > 10.0 && ((base + j) % 64) == 0) {
                    #pragma omp critical(largeblock)
                    {
                        huge_count += 1;
                        huge_sum += av;
                    }
                }
            }
        }

        #pragma omp sections
        {
            #pragma omp section
            {
                double s = 0.0;
                for (int j = 0; j < N; ++j) {
                    s += C[j];
                }
                first_row = s;
            }
            #pragma omp section
            {
                double s = 0.0;
                const double *row = C + (size_t)(rows - 1) * (size_t)N;
                for (int j = 0; j < N; ++j) {
                    s += row[j];
                }
                last_row = s;
            }
        }

        /* Sections already barrier. This one is the handoff before tasks. */
        #pragma omp barrier
    }

    double slot_row[1] = {0.0};
    double slot_max[1] = {0.0};
    double slot_both[1] = {0.0};
    int sentinel = 0;
    #pragma omp parallel num_threads(threads) default(none) \
        shared(first_row, max_abs, slot_row, slot_max, slot_both, sentinel)
    {
        #pragma omp single
        {
            #pragma omp taskgroup
            {
                #pragma omp task depend(out : slot_row[0])
                {
                    slot_row[0] = first_row;
                }
                #pragma omp task depend(out : slot_max[0])
                {
                    slot_max[0] = max_abs;
                }
                #pragma omp task depend(in : slot_row[0], slot_max[0]) depend(out : slot_both[0])
                {
                    slot_both[0] = slot_row[0] + slot_max[0];
                }
            }
            #pragma omp task
            {
                sentinel = 1;
            }
            #pragma omp taskwait
        }
    }

    double scale = fabs(crit_sum) + fabs(red_sum) + 1e-12;
    int sums_agree = fabs(crit_sum - red_sum) / scale <= 1e-8;
    int pack_ok = fabs(slot_row[0] - first_row) <= 1e-8 * (fabs(first_row) + 1.0) &&
                  fabs(slot_max[0] - max_abs) <= 1e-8 * (fabs(max_abs) + 1.0) &&
                  sentinel == 1;

    out->crit_sum = crit_sum;
    out->red_sum = red_sum;
    out->max_abs = max_abs;
    out->first_row = first_row;
    out->last_row = last_row;
    out->packed_row = slot_row[0];
    out->packed_max = slot_max[0];
    out->packed_both = slot_both[0];
    out->large_count = large_count;
    out->huge_count = huge_count;
    out->huge_sum = huge_sum;
    out->stats_ok = sums_agree && pack_ok;
    if (!sums_agree) {
        fprintf(stderr, "checksum mismatch critical=%.6e reduction=%.6e\n", crit_sum, red_sum);
    }
    if (!pack_ok) {
        fprintf(stderr, "task pack mismatch sentinel=%d\n", sentinel);
    }
}

static void broadcast_options(Options *opt, int rank) {
    long long mine[6];
    long long rootv[6];
    mine[0] = opt->N;
    mine[1] = (long long)opt->seed;
    mine[2] = opt->threads;
    mine[3] = opt->schedule;
    mine[4] = opt->chunk;
    mine[5] = opt->trace;
    if (rank == 0) {
        memcpy(rootv, mine, sizeof rootv);
    }
    mpi_check(MPI_Bcast(rootv, 6, MPI_LONG_LONG, 0, MPI_COMM_WORLD), "Bcast(control)");
    for (int i = 0; i < 6; ++i) {
        if (mine[i] != rootv[i]) {
            fprintf(stderr, "rank %d disagreed with root on control field %d\n", rank, i);
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
    }
}

static void memory_guard(int rank, int P, int N, int local_rows) {
    long long mine;
    if (rank == 0) {
        mine = 3LL * (long long)N * (long long)N * (long long)sizeof(double);
        if (N <= FULL_VERIFY_MAX_N) {
            mine += (long long)N * (long long)N * (long long)sizeof(double);
        }
    } else {
        mine = (2LL * (long long)local_rows * (long long)N + (long long)N * (long long)N) *
               (long long)sizeof(double);
    }

    char myname[MPI_MAX_PROCESSOR_NAME];
    int namelen = 0;
    mpi_check(MPI_Get_processor_name(myname, &namelen), "Get_processor_name");
    char *names = (char *)malloc((size_t)P * MPI_MAX_PROCESSOR_NAME);
    if (!names) {
        fprintf(stderr, "failed to allocate hostname buffer\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    mpi_check(MPI_Allgather(myname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, names,
                            MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD),
              "Allgather(hostname)");
    int same_host = 1;
    for (int r = 0; r < P; ++r) {
        if (strncmp(names + (size_t)r * MPI_MAX_PROCESSOR_NAME, myname, MPI_MAX_PROCESSOR_NAME) != 0) {
            same_host = 0;
            break;
        }
    }
    free(names);

    long long total = 0;
    mpi_check(MPI_Allreduce(&mine, &total, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
              "Allreduce(memory)");
    long long avail = read_mem_available_bytes();
    long long limit_src = same_host ? total : mine;
    if (rank == 0) {
        printf("memory_estimate_mib=%.1f available_mib=%.1f same_host=%d\n",
               (double)total / (1024.0 * 1024.0),
               avail > 0 ? (double)avail / (1024.0 * 1024.0) : -1.0, same_host);
        fflush(stdout);
    }
    if (avail > 0 && limit_src > avail * 3 / 4) {
        if (rank == 0) {
            fprintf(stderr,
                    "refusing N=%d P=%d: this run needs about %.1f MiB and only %.1f MiB is available\n",
                    N, P, (double)limit_src / (1024.0 * 1024.0),
                    (double)avail / (1024.0 * 1024.0));
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

static void exchange_and_scatter(double *A, double *A_local, int N, int P, int rank,
                                 int local_rows, int row_start) {
    if (rank == 0) {
        for (int r = 1; r < P; ++r) {
            int lr = 0, rs = 0;
            row_layout(N, P, r, &lr, &rs);
            int pair[2] = {lr, rs};
            mpi_check(MPI_Send(pair, 2, MPI_INT, r, TAG_ROWS, MPI_COMM_WORLD), "Send(rows)");
        }
    } else {
        int pair[2] = {0, 0};
        mpi_check(MPI_Recv(pair, 2, MPI_INT, 0, TAG_ROWS, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                  "Recv(rows)");
        if (pair[0] != local_rows || pair[1] != row_start) {
            fprintf(stderr, "rank %d row assignment mismatch\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
    }

    int equal = (N % P == 0);
    int local_count = mpi_elems(local_rows, N, 0);
    if (equal) {
        int count = mpi_elems(N / P, N, 0);
        if (rank == 0) {
            mpi_check(MPI_Scatter(A, count, MPI_DOUBLE, MPI_IN_PLACE, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD),
                      "Scatter");
        } else {
            mpi_check(MPI_Scatter(NULL, 0, MPI_DOUBLE, A_local, count, MPI_DOUBLE, 0, MPI_COMM_WORLD),
                      "Scatter");
        }
    } else if (rank == 0) {
        int *counts = (int *)malloc((size_t)P * sizeof(int));
        int *displs = (int *)malloc((size_t)P * sizeof(int));
        if (!counts || !displs) {
            fprintf(stderr, "failed to allocate scatter counts\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int r = 0; r < P; ++r) {
            int lr = 0, rs = 0;
            row_layout(N, P, r, &lr, &rs);
            counts[r] = mpi_elems(lr, N, 0);
            displs[r] = mpi_elems(rs, N, 1);
        }
        mpi_check(MPI_Scatterv(A, counts, displs, MPI_DOUBLE, MPI_IN_PLACE, 0, MPI_DOUBLE, 0,
                               MPI_COMM_WORLD),
                  "Scatterv");
        free(counts);
        free(displs);
    } else {
        mpi_check(MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, A_local, local_count, MPI_DOUBLE, 0,
                               MPI_COMM_WORLD),
                  "Scatterv");
    }
}

static void gather_c(double *C, double *C_local, int N, int P, int rank, int local_rows) {
    int equal = (N % P == 0);
    int local_count = mpi_elems(local_rows, N, 0);
    if (equal) {
        int count = mpi_elems(N / P, N, 0);
        if (rank == 0) {
            mpi_check(MPI_Gather(MPI_IN_PLACE, 0, MPI_DOUBLE, C, count, MPI_DOUBLE, 0, MPI_COMM_WORLD),
                      "Gather");
        } else {
            mpi_check(MPI_Gather(C_local, count, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD),
                      "Gather");
        }
    } else if (rank == 0) {
        int *counts = (int *)malloc((size_t)P * sizeof(int));
        int *displs = (int *)malloc((size_t)P * sizeof(int));
        if (!counts || !displs) {
            fprintf(stderr, "failed to allocate gather counts\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int r = 0; r < P; ++r) {
            int lr = 0, rs = 0;
            row_layout(N, P, r, &lr, &rs);
            counts[r] = mpi_elems(lr, N, 0);
            displs[r] = mpi_elems(rs, N, 1);
        }
        mpi_check(MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, C, counts, displs, MPI_DOUBLE, 0,
                              MPI_COMM_WORLD),
                  "Gatherv");
        free(counts);
        free(displs);
    } else {
        mpi_check(MPI_Gatherv(C_local, local_count, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0,
                              MPI_COMM_WORLD),
                  "Gatherv");
    }
}

static void exchange_reports(int rank, int P, const double outb[REPORT_DOUBLES], double *all) {
    MPI_Request send_req = MPI_REQUEST_NULL;
    if (rank == 0) {
        MPI_Request *recvs = (MPI_Request *)malloc((size_t)P * sizeof(MPI_Request));
        if (!recvs) {
            fprintf(stderr, "failed to allocate report requests\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int r = 0; r < P; ++r) {
            mpi_check(MPI_Irecv(all + (size_t)r * REPORT_DOUBLES, REPORT_DOUBLES, MPI_DOUBLE, r,
                                TAG_REPORT, MPI_COMM_WORLD, &recvs[r]),
                      "Irecv(report)");
        }
        mpi_check(MPI_Isend(outb, REPORT_DOUBLES, MPI_DOUBLE, 0, TAG_REPORT, MPI_COMM_WORLD, &send_req),
                  "Isend(report)");
        mpi_check(MPI_Wait(&send_req, MPI_STATUS_IGNORE), "Wait(own report)");
        mpi_check(MPI_Waitall(P, recvs, MPI_STATUSES_IGNORE), "Waitall(reports)");
        free(recvs);
    } else {
        mpi_check(MPI_Isend(outb, REPORT_DOUBLES, MPI_DOUBLE, 0, TAG_REPORT, MPI_COMM_WORLD, &send_req),
                  "Isend(report)");
        mpi_check(MPI_Wait(&send_req, MPI_STATUS_IGNORE), "Wait(report)");
    }
}

int main(int argc, char **argv) {
    int provided = 0;
    int rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (rc != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init_thread failed\n");
        return 1;
    }
    int rank = 0, P = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "MPI_THREAD_FUNNELED unavailable (provided %d)\n", provided);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    Options opt;
    int parsed = parse_options(argc, argv, &opt);
    if (parsed > 0) {
        if (rank == 0) {
            usage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }
    if (parsed < 0) {
        if (rank == 0) {
            usage(argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    broadcast_options(&opt, rank);
    if (P > opt.N) {
        if (rank == 0) {
            fprintf(stderr, "process count %d is larger than N %d\n", P, opt.N);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    int local_rows = 0, row_start = 0;
    row_layout(opt.N, P, rank, &local_rows, &row_start);
    memory_guard(rank, P, opt.N, local_rows);

    int offered_procs = omp_get_num_procs();
    int offered_threads = omp_get_max_threads();
    if (rank == 0) {
        printf("cores=%d processes=%d threads=%d\n", P * opt.threads, P, opt.threads);
        printf("procs_available=%d max_threads_offered=%d mpi_thread_provided=%d schedule=%s\n",
               offered_procs, offered_threads, provided, schedule_name(opt.schedule, opt.chunk));
        printf("scatter_mode=%s\n", (opt.N % P == 0) ? "scatter" : "scatterv");
        if (opt.trace) {
            fprintf(stderr, "trace forces schedule(static) ordered and is outside the quiet benchmark path\n");
        }
        fflush(stdout);
    }

    size_t nn = (size_t)opt.N * (size_t)opt.N;
    size_t local_n = (size_t)local_rows * (size_t)opt.N;
    double *A = NULL;
    double *B = alloc_doubles(nn);
    double *C = NULL;
    double *A_local = NULL;
    double *C_local = NULL;
    if (rank == 0) {
        A = alloc_doubles(nn);
        C = alloc_doubles(nn);
        A_local = A + (size_t)row_start * (size_t)opt.N;
        C_local = C + (size_t)row_start * (size_t)opt.N;
    } else {
        A_local = alloc_doubles(local_n);
        C_local = alloc_doubles(local_n);
    }
    if (!B || !A_local || !C_local || (rank == 0 && (!A || !C))) {
        fprintf(stderr, "rank %d allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (rank == 0) {
        unsigned long long state = opt.seed;
        lcg_fill(A, nn, &state);
        lcg_fill(B, nn, &state);
    }

    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "Barrier");
    double t0 = MPI_Wtime();
    exchange_and_scatter(A, A_local, opt.N, P, rank, local_rows, row_start);
    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "Barrier");
    double t_scatter = MPI_Wtime() - t0;

    int bcast_count = mpi_elems(opt.N, opt.N, 0);
    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "Barrier");
    t0 = MPI_Wtime();
    mpi_check(MPI_Bcast(B, bcast_count, MPI_DOUBLE, 0, MPI_COMM_WORLD), "Bcast(B)");
    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "Barrier");
    double t_bcast = MPI_Wtime() - t0;

    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "Barrier");
    t0 = MPI_Wtime();
    MatmulInfo info;
    matmul_block(A_local, B, C_local, local_rows, opt.N, opt.threads, opt.schedule, opt.chunk,
                 opt.trace, &info);
    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "Barrier");
    double t_compute = MPI_Wtime() - t0;

    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "Barrier");
    t0 = MPI_Wtime();
    gather_c(C, C_local, opt.N, P, rank, local_rows);
    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "Barrier");
    double t_gather = MPI_Wtime() - t0;
    double t_total = t_scatter + t_bcast + t_compute + t_gather;

    LocalStats stats;
    local_stats(C_local, row_start, local_rows, opt.N, opt.threads, &stats);

    double outb[REPORT_DOUBLES];
    outb[0] = (double)local_rows;
    outb[1] = (double)info.team_seen;
    outb[2] = (double)info.last_row;
    outb[3] = stats.crit_sum;
    outb[4] = stats.red_sum;
    outb[5] = stats.max_abs;
    outb[6] = stats.first_row;
    outb[7] = stats.last_row;
    outb[8] = (double)stats.large_count;
    outb[9] = stats.packed_row;
    outb[10] = stats.packed_max;
    double *reports = NULL;
    if (rank == 0) {
        reports = (double *)calloc((size_t)P * REPORT_DOUBLES, sizeof(double));
        if (!reports) {
            fprintf(stderr, "failed to allocate reports\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    exchange_reports(rank, P, outb, reports);

    double neighbor_time = 0.0;
    int next = (rank + 1) % P;
    int prev = (rank + P - 1) % P;
    mpi_check(MPI_Sendrecv(&t_compute, 1, MPI_DOUBLE, next, TAG_RING, &neighbor_time, 1, MPI_DOUBLE,
                           prev, TAG_RING, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
              "Sendrecv");

    double *all_times = (double *)malloc((size_t)P * sizeof(double));
    double *all_neighbors = (double *)malloc((size_t)P * sizeof(double));
    if (!all_times || !all_neighbors) {
        fprintf(stderr, "failed to allocate timing buffers\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    mpi_check(MPI_Allgather(&t_compute, 1, MPI_DOUBLE, all_times, 1, MPI_DOUBLE, MPI_COMM_WORLD),
              "Allgather(time)");
    mpi_check(MPI_Allgather(&neighbor_time, 1, MPI_DOUBLE, all_neighbors, 1, MPI_DOUBLE, MPI_COMM_WORLD),
              "Allgather(neighbor)");

    double red_local = stats.red_sum;
    double crit_local = stats.crit_sum;
    double max_local = stats.max_abs;
    double red_global = 0.0, crit_global = 0.0, max_global = 0.0;
    mpi_check(MPI_Reduce(&red_local, &red_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD),
              "Reduce(checksum)");
    mpi_check(MPI_Reduce(&crit_local, &crit_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD),
              "Reduce(critical checksum)");
    mpi_check(MPI_Reduce(&max_local, &max_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
              "Reduce(max)");

    double min_time = 0.0;
    mpi_check(MPI_Allreduce(&t_compute, &min_time, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD),
              "Allreduce(min time)");

    int owns = 0;
    double local_sample = 0.0;
    int ns = sample_count(opt.N);
    for (int s = 0; s < ns; ++s) {
        int row = sample_row_index(opt.N, s);
        if (row < row_start || row >= row_start + local_rows) {
            continue;
        }
        owns = 1;
        int li = row - row_start;
        double err = row_error(A_local + (size_t)li * (size_t)opt.N, B,
                               C_local + (size_t)li * (size_t)opt.N, opt.N);
        if (err > local_sample) {
            local_sample = err;
        }
    }

    MPI_Comm sub = MPI_COMM_NULL;
    mpi_check(MPI_Comm_split(MPI_COMM_WORLD, owns ? 0 : MPI_UNDEFINED, rank, &sub), "Comm_split");
    double sample_err = 0.0;
    if (owns) {
        mpi_check(MPI_Allreduce(&local_sample, &sample_err, 1, MPI_DOUBLE, MPI_MAX, sub),
                  "Allreduce(sample)");
        mpi_check(MPI_Comm_free(&sub), "Comm_free");
    }

    double full_err = -1.0;
    if (rank == 0 && opt.N <= FULL_VERIFY_MAX_N) {
        double *Cref = alloc_doubles(nn);
        if (!Cref) {
            fprintf(stderr, "failed to allocate reference matrix\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MatmulInfo ignored;
        matmul_block(A, B, Cref, opt.N, opt.N, 1, SCHED_STATIC, 0, 0, &ignored);
        full_err = max_abs_diff(C, Cref, nn);
        free(Cref);
    }

    double direct_checksum = 0.0;
    if (rank == 0) {
        direct_checksum = strided_checksum(C, 0, opt.N, opt.N);
    }

    int local_pass = stats.stats_ok ? 1 : 0;
    if (owns && local_sample > VERIFY_ABS_TOL) {
        local_pass = 0;
    }
    if (rank == 0 && full_err >= 0.0 && full_err > VERIFY_ABS_TOL) {
        local_pass = 0;
    }
    if (rank == 0) {
        double scale = fabs(direct_checksum) + fabs(red_global) + 1e-12;
        if (fabs(direct_checksum - red_global) / scale > 1e-8 ||
            fabs(direct_checksum - crit_global) / scale > 1e-8) {
            local_pass = 0;
            fprintf(stderr, "global checksum mismatch direct=%.6e reduction=%.6e critical=%.6e\n",
                    direct_checksum, red_global, crit_global);
        }
    }
    int all_pass = 0;
    mpi_check(MPI_Allreduce(&local_pass, &all_pass, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD),
              "Allreduce(pass)");

    if (rank == 0) {
        double reported_err = full_err >= 0.0 ? full_err : sample_err;
        double gflops = (t_compute > 0.0)
                            ? ((2.0 * (double)opt.N) * (double)opt.N * (double)opt.N) / t_compute / 1e9
                            : 0.0;
        printf("team_seen=%d last_row=%d clock=MPI_Wtime\n", info.team_seen, info.last_row);
        printf("t_scatter=%.6f t_bcast=%.6f t_compute=%.6f t_gather=%.6f t_total=%.6f\n",
               t_scatter, t_bcast, t_compute, t_gather, t_total);
        printf("gflops=%.6f checksum=%.6e max_abs_err=%.6e sample_err=%.6e min_compute=%.6f\n",
               gflops, direct_checksum, reported_err, sample_err, min_time);
        printf("checksum_reduction=%.6e checksum_critical=%.6e entry_max=%.6e\n",
               red_global, crit_global, max_global);
        for (int r = 0; r < P; ++r) {
            const double *b = reports + (size_t)r * REPORT_DOUBLES;
            printf("rank %d rows=%.0f team=%.0f last_row=%.0f crit=%.6e red=%.6e max=%.6e "
                   "first=%.6e last=%.6e large=%.0f packed_row=%.6e neighbor_time=%.6f compute_time=%.6f\n",
                   r, b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], all_neighbors[r],
                   all_times[r]);
        }
        printf("VERIFY %s\n", all_pass ? "PASS" : "FAIL");
        printf("csv,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6e,%.6e,%s,hybrid,%d\n",
               opt.N, P, opt.threads, t_scatter, t_bcast, t_compute, t_gather, t_total, gflops,
               reported_err, direct_checksum, schedule_name(opt.schedule, opt.chunk), opt.oversubscribe);
        fflush(stdout);
    }

    free(all_times);
    free(all_neighbors);
    free(reports);
    if (rank == 0) {
        free(A);
        free(C);
    } else {
        free(A_local);
        free(C_local);
    }
    free(B);
    MPI_Finalize();
    return all_pass ? 0 : 1;
}
