# Parallel Dense Matrix Multiplication

Hybrid MPI + OpenMP multiplication of two dense square matrices, `C = A × B`, in double precision. The default problem is **8192 × 8192** (about 512 MiB per matrix, about 1.10 × 10^12 floating-point operations). N, the process count, and the thread count are all runtime arguments, so the same binary runs on this machine and on a larger one.

## Layout

| Path | Role |
|---|---|
| `src/common.c` | Seeded generator, i-k-j kernel, strided checksum |
| `src/sequential.c` | One-thread baseline, timed with `omp_get_wtime` |
| `src/hybrid.c` | MPI row distribution and the OpenMP team inside each rank |
| `scripts/run.sh` | One hybrid run from a core budget or an explicit P × T split |
| `scripts/run_benchmarks.sh` | Strong-scaling sweep from `--cores` (default `nproc`) |
| `scripts/verify.sh` | Correctness checks |
| `scripts/plot_results.py` | Time, speedup, efficiency, and breakdown figures |
| `scripts/check_kernel.py` | Independent Python copy of the generator and the product |
| `REPORT.md` | Written report, including the measured numbers |
| `results/timings.csv` | One row per measured run |

## Build

The programs are C11. Both compile with `-O3 -std=c11 -fopenmp`. `-ffast-math` is not used.

```bash
sudo apt install build-essential libopenmpi-dev python3-venv
make
```

`make` uses `gcc` for the sequential binary and `mpicc` for the hybrid binary. If `mpicc` is not on `PATH` but a user-local OpenMPI exists at `~/.local/openmpi`, `scripts/env.sh` adds it. That is how this machine was built: `sudo` asked for a password, so OpenMPI 4.1.6 was configured with `--prefix=$HOME/.local/openmpi`. A lab image that already has `mpicc` does not need that prefix.

Figures need matplotlib in the project virtualenv:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

From the repository root, inside WSL or any Linux host:

```bash
./scripts/run.sh --cores 10 --size 8192
```

That uses **exactly 10 cores**. The script factors 10 as the pair closest to a square and, on a tie, keeps the smaller process count so B is copied fewer times. 10 becomes **2 processes × 5 threads**. 16 becomes 4 × 4. A prime budget such as 7 becomes 1 × 7. An explicit split is:

```bash
./scripts/run.sh --procs 5 --threads 2 --size 8192
```

If `--cores` is also passed, `procs × threads` must equal it. The script refuses a budget above `nproc` unless `--allow-oversubscribe` is set. Binding is `mpirun --map-by socket:PE=$T --bind-to core` (a core cannot hold more than one of those PEs, so `core:PE=T` is rejected when T > 1). Each rank is started with `OMP_NUM_THREADS` set to T. After binding, `omp_get_num_procs` reports the cores given to that rank. The program itself does **not** default to “every core on the machine”: with no `--threads` and no `OMP_NUM_THREADS`, each rank uses 1 thread. Otherwise `mpirun -np 10` on a 16-core host would start 10 × 16 threads.

Other flags: `--seed` (default 42), `--schedule static|static,C|dynamic[,C]|guided[,C]|runtime`, `--trace`, `--verify`. `--trace` forces the static schedule and prints a few row ids in loop order on stderr. `--verify` is accepted and documented; the checks run on every launch. For `runtime`, export `OMP_SCHEDULE` (the benchmark script sets it to `static`).

The first stdout line of a hybrid run is the core budget, for example `cores=10 processes=2 threads=5`. A later line reports `team_seen`. If the block is large enough to parallelize and the team is smaller than requested, the program warns on stderr.

### Distribution

- P divides N: `MPI_Scatter` of equal row blocks of A and `MPI_Gather` of C. This is the exam program.
- P does not divide N (10 processes, N = 8192): `MPI_Scatterv` / `MPI_Gatherv`. The first `N % P` ranks own one extra row. Ranks learn their row count from `MPI_Send` / `MPI_Recv`; only the root holds the full count vector.
- P > N is rejected.
- B is `MPI_Bcast` to every rank. Root fills A and B from a documented 64-bit LCG (not `rand()`), seed 42, values in [-1, 1].

Before allocating, the program prints the byte estimate and refuses the run when a single host would have to hold more than 75% of `MemAvailable`. At N = 8192 the root holds A, B, and C (1.5 GiB) and every rank holds its own B.

## Correctness

`./scripts/verify.sh` runs:

- sequential N = 128 and N = 256
- hybrid N = 128 and N = 512 with 2 processes × 2 threads (oversubscribed on a 3-core host; the check is about the answer, not the speed)
- hybrid N = 128 with 1 × 1, checksum compared with the sequential binary
- hybrid N = 1024 with 1 × 1 (8-row spot check)
- hybrid N = 100 with 3 processes, which forces `MPI_Scatterv`

For N ≤ 512 the root also multiplies A and B again with the one-thread kernel and compares every entry. The pass threshold is an absolute error of 1e-6. Larger N recomputes 8 spread-out rows (row 0, N/8, …) with an i-j-k dot product, which is the same arithmetic as the i-k-j kernel. `scripts/check_kernel.py` repeats the generator and the product in Python; for N = 32 its checksum matches the C binary (`6.996893e-03`).

## Benchmarks and figures

```bash
./scripts/run_benchmarks.sh              # --cores defaults to nproc
./scripts/run_benchmarks.sh --cores 10   # same sweep on a larger machine
.venv/bin/python scripts/plot_results.py
```

For a budget K the sweep is:

- sequential baseline (1 thread), three repeats at N = 1024, one at N = 2048
- pure OpenMP: P = 1, T = 1..K
- pure MPI: T = 1, P = 1..K
- the default factor pair of each c = 1..K, when that pair is not already in the two lines above
- one oversubscribed point: that pair of K with one extra thread per rank, at N = 1024
- N = 4096 for the sequential binary, pure OpenMP at K, and the hybrid pair when it differs
- schedule comparison at N = 1024 on the default pair: `static`, `static,1`, `dynamic,1`, `guided,1`, `runtime` with `OMP_SCHEDULE=static`
- one attempt at N = 8192 (sequential, then the default hybrid pair). A refusal is recorded in `results/n8192_status.txt` instead of swapping

Speedup is sequential total time divided by hybrid total time. Efficiency divides that by P × T. The headline time includes scatter, broadcast, compute, and gather. Generation, the stats phase, and the spot check are outside the timer. `results/summary.md` also fits an Amdahl serial fraction.

CSV columns:

`N,P,threads,t_scatter,t_bcast,t_compute,t_gather,t_total,gflops,max_abs_err,checksum,schedule,program,oversubscribe`

Chunked schedule names are stored with an underscore (`static_1`, `dynamic_1`, `guided_1`) so the schedule column stays one field. The command-line flag still uses a comma: `--schedule static,1`.

Figures in `results/figures/`: `time_vs_threads.png`, `time_vs_processes.png`, `speedup.png`, `efficiency.png`, `time_breakdown.png`.

## Syllabus map

The notes are the PDC syntax handbook (OpenMP + MPI). The timed multiply is the handbook matrix program: scatter rows of A, broadcast B, an OpenMP team over output rows, gather C. The other calls still produce a printed number, a timer, or the pass/fail line. They sit in setup or in the stats phase after `t_compute`, so they do not inflate the flop rate.

### Used

| Construct | Where |
|---|---|
| `gcc -fopenmp`, `omp_get_wtime` | `src/sequential.c`, `Makefile` |
| `mpicc -fopenmp`, `mpirun -np` | `Makefile`, `scripts/run.sh` |
| `omp parallel`, `num_threads`, `if`, `default(none)`, `shared`, `private`, `firstprivate` | `matmul_block` in `src/common.c` |
| `omp_set_num_threads`, `omp_set_dynamic(0)` | `matmul_block` |
| `omp for` inside one parallel region, `collapse(2)`, `schedule(static)` | zeroing and the product in `matmul_block` |
| `schedule(static,chunk)`, `schedule(dynamic,chunk)`, `schedule(guided,chunk)`, `schedule(runtime)` | `matmul_block`, selected by `--schedule` |
| `omp simd` | inner j loop, `row_product` |
| `lastprivate` | last row covered, `matmul_block` |
| `master`, `omp_get_num_threads`, `omp_get_thread_num` | team-size check and `--trace` |
| `omp_get_max_threads`, `omp_get_num_procs` | startup line in both drivers |
| `ordered` | `--trace` path in `matmul_block` |
| `omp for nowait`, `critical(checksum)` | strided checksum in `local_stats` |
| `atomic update`, `critical(largeblock)` | strided entry counters in `local_stats` |
| `reduction(+:)`, `reduction(max:)` | the same checksum and the max absolute entry; they must agree with the critical checksum |
| `sections` / `section` | first-row sum and last-row sum |
| `barrier` | between the section phase and the task phase |
| `single`, `task`, `taskgroup`, `task depend`, `taskwait` | packing those stats in `local_stats` |
| `MPI_Init_thread` (`MPI_THREAD_FUNNELED`), `MPI_Comm_rank`, `MPI_Comm_size`, `MPI_Finalize` | `main` in `src/hybrid.c` |
| `MPI_Bcast` | control block and matrix B |
| `MPI_Scatter`, `MPI_Gather` | equal row blocks |
| `MPI_Scatterv`, `MPI_Gatherv` | unequal row blocks |
| `MPI_Send`, `MPI_Recv` | per-rank row count and start |
| `MPI_Isend`, `MPI_Irecv`, `MPI_Wait`, `MPI_Waitall` | per-rank report after the multiply |
| `MPI_Sendrecv` | ring exchange of compute time |
| `MPI_Reduce` with `MPI_SUM` and `MPI_MAX` | global checksum and max entry |
| `MPI_Allreduce` with `MPI_MIN` and `MPI_LAND` | fastest rank, and “every rank passed” |
| `MPI_Allgather` | per-rank times |
| `MPI_Barrier`, `MPI_Wtime` | the four timed regions |
| `MPI_Abort` | bad arguments, P > N, allocation failure, failed collective |
| `MPI_Comm_split`, `MPI_Comm_free` | ranks that own a sample row, then an `MPI_Allreduce` of their error |
| `MPI_DOUBLE`, `MPI_INT`, `MPI_LONG_LONG` | payloads |

`MPI_Init` is what the notes use when the process has no extra threads. This program calls `MPI_Init_thread` and asks for `MPI_THREAD_FUNNELED`, because OpenMP threads exist and only the main thread calls MPI.

### Studied, not used

| Construct | Why it is the wrong tool here |
|---|---|
| `MPI_Alltoall` | No rank sends a distinct block to every other rank. B is one broadcast. An all-to-all of identical values would be a slower allgather. |
| `reduction(*:)` and `reduction(-:)` | A product or a difference is not a result of matrix multiplication. The subtraction-reduction trap in the notes (private contributions are combined by addition) is why that clause is easy to misread, and it is not reimplemented. |
| `MPI_PROD` | Same reason as `reduction(*:)`. |
| `MPI_LOR` | The pass flag is “every rank passed”, which is `MPI_LAND`. A logical OR would pass when a single rank passed. |
| `masked` | Same job as `master`. The quizzes use `master`, so that is the directive in `matmul_block`. |

Speedup = serial time / parallel time. Efficiency = speedup / (P × T). Amdahl is `1 / (s + (1-s)/C)` with C = P × T. Gustafson is `C - s(C-1)`. The report fits s from the measured times. Gustafson is discussed and not fitted, because N stays fixed: this is strong scaling.
