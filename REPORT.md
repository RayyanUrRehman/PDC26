# Parallel Dense Matrix Multiplication

Hybrid MPI + OpenMP multiplication of two dense square double-precision matrices. The design point is N = 8192. The same binary takes the matrix size, the process count, and the thread count at run time.

## 1. Problem

Compute C = A × B for square matrices of order N. Each entry is

C[i][j] = sum over k of A[i][k] * B[k][j].

That is 2 N³ floating-point operations. At N = 8192 this is 1.0995 × 10¹² operations. One matrix is N² × 8 bytes = 512 MiB, so A, B, and C together are 1.5 GiB. No external dataset is used. The root fills A and then B from a 64-bit linear congruential generator, seed 42, with values in [-1, 1]. The recurrence is `state = state * 6364136223846793005 + 1`, and the top 53 bits are mapped onto [-1, 1]. `rand()` is not used, because its sequence is not the same across C libraries.

## 2. Why hybrid MPI and OpenMP

MPI processes do not share memory, so they are the way to spread rows across a machine or a cluster. OpenMP threads inside one process share B, which every output row needs in full. A pure MPI run would copy B once per process. A pure OpenMP run cannot leave the node. The hybrid program does both: `mpirun -np P` starts the processes, and each process multiplies its own rows with T threads.

The core budget is not compiled in. `scripts/run.sh --cores K` factors K as P × T, choosing the pair closest to a square and, on a tie, the smaller P, so B is copied fewer times. Ten cores become 2 processes × 5 threads. Sixteen become 4 × 4. A prime budget such as 3 or 7 becomes one process and that many threads, because no other pair multiplies to exactly that number. `--procs` and `--threads` override the split. The program does not silently start one OpenMP thread per core on every rank. Without `--threads` and without `OMP_NUM_THREADS`, each rank uses one thread.

## 3. Design

Matrices are row-major arrays of `double`, indexed as `i * N + j`. The kernel is the i-k-j loop: for each local row i and each k, stream row k of B into row i of C. The arithmetic is the same sum as the textbook i-j-k product. i-k-j keeps the writes to C and the reads of B sequential. Both the sequential binary and the hybrid binary call this kernel, so a speedup is not an artifact of a different loop nest.

Distribution follows the handbook matrix program.

- When P divides N, the root uses `MPI_Scatter` for equal row blocks of A and `MPI_Gather` for C.
- When P does not divide N, it uses `MPI_Scatterv` and `MPI_Gatherv`. The first `N % P` ranks own one extra row. Ten processes and N = 8192 is this case (8192 = 10 × 819 + 2).
- The root sends each worker its row count and starting row with `MPI_Send` / `MPI_Recv`. Only the root allocates the full count and displacement arrays.
- B is sent with `MPI_Bcast`. Every rank needs every row of B to finish its own rows of C.
- P > N is rejected with `MPI_Abort`.

The root keeps its own rows in place (`MPI_IN_PLACE`), so it does not allocate a second copy of A or C. Workers allocate only their block of A, their block of C, and a full B.

`MPI_Init_thread` requests `MPI_THREAD_FUNNELED`. OpenMP threads do the arithmetic. Only the main thread calls MPI, and it does so outside the parallel region. The notes' `MPI_Init` is the call for a process with no extra threads. A hybrid program has to name the thread level. On this OpenMPI build the provided level was `MPI_THREAD_FUNNELED`.

Inside each rank the team is created once. The region is `#pragma omp parallel num_threads(T) if(rows > 32) default(none)`, with explicit `shared`, `private`, and `firstprivate`. The `if` clause leaves a tiny block on one thread. `omp_set_dynamic(0)` stops the runtime from changing the team size. Two worksharing loops follow: `collapse(2)` zeroes the local C, then `omp for` does the product. The inner j loop is `omp simd`. `schedule(static)` is the default. `--schedule` also selects `static,chunk`, `dynamic,chunk`, `guided,chunk`, and `runtime` (the last one reads `OMP_SCHEDULE`). `lastprivate` records the last row index. `master` stores `omp_get_num_threads()` so the run can warn if the team was smaller than requested.

A stats phase runs after the compute timer. It is real work that is printed, and it is where the rest of the syllabus constructs sit: a `nowait` loop plus `critical(checksum)`, a `reduction(+:)` of the same strided checksum, `reduction(max:)`, `atomic update`, a second named critical, `sections` for the first and last local rows, an explicit `barrier`, then `single` / `task` / `taskgroup` / `task depend` / `taskwait` to pack those values. `--trace` adds `ordered` and is off for the timings.

After that, each rank `MPI_Isend`s a short report and the root completes it with `MPI_Irecv`, `MPI_Wait`, and `MPI_Waitall`. `MPI_Sendrecv` moves each rank's compute time around a ring. `MPI_Reduce` combines the checksum (`MPI_SUM`) and the max entry (`MPI_MAX`). `MPI_Allreduce` publishes the fastest compute time (`MPI_MIN`) and whether every rank passed (`MPI_LAND`). `MPI_Allgather` collects the per-rank times. Sample rows (row 0, N/8, …) usually fall on more than one rank. Those ranks join a subcommunicator through `MPI_Comm_split`; the others pass `MPI_UNDEFINED` and must not call the following collective. The subcommunicator reduces the sample error and is released with `MPI_Comm_free`.

The four timed regions, each wrapped in `MPI_Barrier` and `MPI_Wtime`, are scatter, broadcast, compute, and gather. Filling the matrices, the stats phase, and the spot check are outside those timers. The sequential binary is timed with `omp_get_wtime` and is forced to one thread. Both clocks are wall clocks.

Before allocating, the program estimates the bytes and reads `MemAvailable`. On a single host the sum across ranks has to fit in 75% of that figure, otherwise the job aborts instead of swapping.

## 4. Complexity

Each rank owns about N/P rows and does O(N³/P) arithmetic. The broadcast of B moves O(N²) data, and the scatter and gather move O(N²) together. Threads do not change the communication volume. They divide the local rows.

Speedup is sequential total time divided by hybrid total time. Efficiency is that speedup divided by P × T. Amdahl's law is `1 / (s + (1 - s) / C)` with C = P × T. Gustafson's law is `C - s (C - 1)`. This experiment holds N fixed, so it is strong scaling and Amdahl is the relevant model. Gustafson would require growing N with the machine. Section 9 says what the fitted s actually means for these runs.

## 5. Machine and software

The host is an AMD PRO A12-9800B (4 cores, 4 threads) with 6.93 GB of RAM. The build and the timings were done in WSL Ubuntu, where `nproc` reports 3 and `MemAvailable` during the runs was about 3.4 GiB. The compiler is GCC 13.3.0. OpenMPI 4.1.6 was built from source into `~/.local/openmpi` because `sudo` on this machine asks for a password; `scripts/env.sh` puts that `mpicc` on `PATH` when there is no system one. Flags are `-O3 -std=c11 -fopenmp`. `-ffast-math` is not used. Figures were drawn with matplotlib in `.venv`.

Binding is `mpirun --map-by socket:PE=T --bind-to core`. Mapping by `core:PE=T` is rejected when T > 1, because one core cannot supply T processing elements. After binding, `omp_get_num_procs` reports the cores given to that rank, not the whole machine.

## 6. Correctness

For N ≤ 512 the root multiplies A and B again with the one-thread kernel and compares every entry. For larger N, including 8192, it recomputes eight spread-out rows with an i-j-k dot product and compares those entries. The pass threshold is an absolute error of 1e-6. Both paths also check that the `critical` checksum and the `reduction(+:)` checksum agree, and that the task pack reproduces the row sum and the max entry.

Every timed run in `results/timings.csv` printed `VERIFY PASS` with max absolute error 0. The checksum of a given N is the same on the sequential binary and on every hybrid configuration. At N = 8192 that shared checksum is 1.130754 × 10⁴. An independent Python copy of the generator and the product, `scripts/check_kernel.py`, matches the C checksum at N = 32 (`6.996893e-03`). N = 100 with 3 processes passed through `MPI_Scatterv`, which is the path used when P does not divide N.

## 7. Methodology

The sequential reference is always one process and one thread. Hybrid speedup uses that binary's total time, not the hybrid 1×1 time, except where the text says otherwise. N = 1024 was repeated three times and the median is reported. N = 2048, 4096, and 8192 were run once. The sweep for a budget K, here K = 3, is pure OpenMP (P = 1, T = 1..K), pure MPI (T = 1, P = 1..K), and the default factor pair of each total from 1 to K when that pair is not already in those two lines. Because 3 is prime, that factor pair is 1 × 3, so the automatic "hybrid" point on this host is pure OpenMP. The MPI runs are the separate T = 1 line. One oversubscribed point uses four threads on three cores. The schedule comparison is N = 1024, one process, three threads.

GFLOP/s uses 2 N³ / t_compute / 10⁹.

## 8. Results

Medians at N = 1024. One repeat otherwise. Communication is scatter + broadcast + gather. Speedup is against the sequential binary.

| N | configuration | t_compute (s) | t_comm (s) | t_total (s) | GFLOP/s | speedup | efficiency |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1024 | sequential | 1.802 | 0 | 1.802 | 1.19 | 1.00 | 1.00 |
| 1024 | 1 × 1 | 1.779 | 0.000 | 1.779 | 1.21 | 1.01 | 1.01 |
| 1024 | 1 × 2 | 1.338 | 0.000 | 1.338 | 1.61 | 1.35 | 0.67 |
| 1024 | 2 × 1 | 1.351 | 0.020 | 1.374 | 1.59 | 1.31 | 0.66 |
| 1024 | 1 × 3 | 1.301 | 0.000 | 1.302 | 1.65 | 1.38 | 0.46 |
| 1024 | 3 × 1 | 1.295 | 0.031 | 1.329 | 1.66 | 1.36 | 0.45 |
| 1024 | 1 × 4, oversubscribed | 1.767 | 0.000 | 1.767 | 1.22 | 1.02 | 0.25 |
| 2048 | sequential | 18.385 | 0 | 18.385 | 0.93 | 1.00 | 1.00 |
| 2048 | 1 × 1 | 14.631 | 0.000 | 14.631 | 1.17 | 1.26 | 1.26 |
| 2048 | 1 × 2 | 10.650 | 0.000 | 10.650 | 1.61 | 1.73 | 0.86 |
| 2048 | 2 × 1 | 10.820 | 0.151 | 10.970 | 1.59 | 1.68 | 0.84 |
| 2048 | 1 × 3 | 10.208 | 0.000 | 10.208 | 1.68 | 1.80 | 0.60 |
| 2048 | 3 × 1 | 9.965 | 0.198 | 10.163 | 1.72 | 1.81 | 0.60 |
| 4096 | sequential | 115.184 | 0 | 115.184 | 1.19 | 1.00 | 1.00 |
| 4096 | 1 × 3 | 81.514 | 0.000 | 81.514 | 1.69 | 1.41 | 0.47 |
| 8192 | sequential | 928.319 | 0 | 928.319 | 1.18 | 1.00 | 1.00 |
| 8192 | 1 × 3 | 722.210 | 0.000 | 722.210 | 1.52 | 1.29 | 0.43 |

The N = 2048 one-process hybrid run was faster than the sequential binary (14.63 s against 18.39 s). That point is a single run, so part of the 1.26 speedup is run-to-run variation. At N = 1024, where there are three repeats, 1 × 1 and the sequential binary agree to about 1%.

Schedule comparison at N = 1024, 1 × 3, one run except the default static median:

| schedule | t_total (s) | speedup |
|---|---:|---:|
| static (default chunk) | 1.302 | 1.38 |
| static, chunk 1 | 2.085 | 0.86 |
| dynamic, chunk 1 | 1.575 | 1.14 |
| guided, chunk 1 | 1.584 | 1.14 |
| runtime, `OMP_SCHEDULE=static` | 1.707 | 1.06 |

Figures, all from these measurements:

- `results/figures/time_vs_threads.png` — one process, N = 1024 and 2048
- `results/figures/time_vs_processes.png` — one thread per process
- `results/figures/speedup.png` — speedup against cores used
- `results/figures/efficiency.png`
- `results/figures/time_breakdown.png` — compute against communication at N = 2048

N = 8192 did run. The estimate was 1536 MiB against about 3412 MiB available, the allocation was accepted, and both the sequential run and the 1 × 3 run finished and matched. A ten-process run of the same N would replicate B ten times and would be refused on this host by the memory check. That is the right outcome: the binary still accepts `--cores 10` on a machine that has the RAM.

## 9. Discussion

The best speedup on the three cores WSL can see is about 1.8, at N = 2048, and it falls to 1.41 at N = 4096 and 1.29 at N = 8192. The second thread helps. The third adds little. At N = 2048, 1 × 2 takes 10.65 s and 1 × 3 takes 10.21 s.

That shape is the memory system, not a forgotten serial loop. In i-k-j, each of the N output rows reads all of B. B at N = 8192 is 512 MiB, and this processor has 2 MB of L2 and no L3, so B is read from DRAM once per row. Extra threads issue more of those reads through the same memory bus. They do not multiply the flop rate by three. Communication is a small slice of the same picture: at N = 2048 with three processes it is 0.20 s out of 10.16 s, about 2%. At N = 8192 the timed run is one process, so scatter and broadcast are local and the communication time rounds to zero. The orange bars in the breakdown figure are visible only on the multi-process bars, and they are thin.

Fitting Amdahl's s from `s = (C T(C) - T(1)) / (T(1) (C - 1))` gives about 0.33 at N = 2048 with three threads and about 0.67 at N = 8192. The communication share of those runs is about 0, so this s is not a serial fraction left in the source. It is the formula's way of saying the parallel kernel itself did not speed up by C. Using that s in Gustafson's formula at C = 3 would predict a speedup near 1.67 for the 8192 case. We did not measure that, because N was not increased with the core count.

`static` with the default chunk is the fastest schedule. A static chunk of 1 is the slowest of the five and is slower than one thread: handing out single rows round-robin breaks the contiguous rows each thread would otherwise own. `dynamic` and `guided` are in between. The rows cost the same, so the extra scheduling does not pay for itself. Four threads on three cores (the oversubscribed point) takes 1.77 s at N = 1024, which is slower than three threads (1.30 s) and about the same as one thread. More workers than cores do not help.

One process with three threads and three processes with one thread are close at N = 2048 (10.21 s and 10.16 s). Copying B three times and paying the broadcast did not beat sharing B inside one process, and it also did not lose by much at this size. On a larger machine the tradeoff moves: more processes copy B again, which is why the default factor pair prefers fewer processes when the two factors are equally balanced.

The N = 8192 result is a real end-to-end run of the assigned size, on the three cores this WSL instance exposes, with a matching checksum and a passing spot check. It is not a 10-core hybrid measurement. That measurement is the same command with `--cores 10` on a host that has ten cores and enough RAM for the replicated B.

## 10. Reproduce, and questions worth being asked

From a Linux shell with `mpicc` on `PATH`, or from WSL on this machine after `scripts/env.sh`:

```bash
make
./scripts/verify.sh
./scripts/run_benchmarks.sh
.venv/bin/python scripts/plot_results.py
```

`./scripts/run.sh --cores 10 --size 8192` is the ten-core run. `./scripts/run_benchmarks.sh --cores 10` rebuilds the curves. Full stdout of the sweep on this machine is under `results/logs/`. The CSV is `results/timings.csv`.

A few questions the design is meant to answer directly:

- **Why split A by rows, not by columns?** A row of C depends on one row of A and on every column of B. Row blocks of A are contiguous in memory, so `MPI_Scatter` can hand each rank a single block, and that rank can finish its rows of C without another word from anyone about A. Column blocks of A would make every entry of C depend on every rank.
- **Why broadcast B instead of splitting it?** Every local row uses every row of B. Splitting B would add an all-to-all or a pipelined shift. The assigned algorithm copies B once. The cost is memory, one full B per process, and it is why the memory check exists.
- **What does `MPI_THREAD_FUNNELED` mean?** The process may have many threads. Only the thread that called `MPI_Init_thread` calls MPI. The OpenMP workers stay inside the multiply.
- **What is the difference between speedup and efficiency?** Speedup is how many times faster the parallel run is than the one-thread baseline. Efficiency divides that by the number of workers, here P × T. A speedup of 1.8 on 3 cores is an efficiency of 0.6: the run is faster, and each added core returned less than one core's worth of the baseline.
- **Why does i-k-j match i-j-k?** Both compute the same sum of N products. Only the order of the additions changes. The spot check is an i-j-k dot product compared with the i-k-j kernel, and the absolute error on these runs was 0 at the reported precision, including eight rows at N = 8192.
- **Why was the three-core run 1 × 3 rather than a mix of processes and threads?** 3 is prime. The only factor pairs are 1 × 3 and 3 × 1, and the rule keeps the smaller process count. The 3 × 1 line was still measured, on purpose, as pure MPI.
- **Which syllabus calls are on the timer?** Scatter or scatterv, broadcast, the parallel team, and gather. Reductions, tasks, the ring `MPI_Sendrecv`, and the sample-row subcommunicator run after `t_compute`. They are in the program because they do a real check or a real report, and they are not part of the flop rate.
