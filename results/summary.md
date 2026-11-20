# Timing summary

Medians. Speedup uses the sequential binary for the same N.

| N | P | T | schedule | over | cores | repeats | t_compute (s) | t_comm (s) | t_total (s) | GFLOP/s | speedup | efficiency | max abs err |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 1 | static | 0 | 1 | 3 | 1.7793 | 0.0000 | 1.7793 | 1.207 | 1.012 | 1.012 | 0.000e+00 |
| 1024 | 1 | 2 | static | 0 | 2 | 3 | 1.3378 | 0.0000 | 1.3378 | 1.605 | 1.347 | 0.673 | 0.000e+00 |
| 1024 | 2 | 1 | static | 0 | 2 | 3 | 1.3510 | 0.0201 | 1.3741 | 1.590 | 1.311 | 0.656 | 0.000e+00 |
| 1024 | 1 | 3 | dynamic_1 | 0 | 3 | 1 | 1.5749 | 0.0000 | 1.5749 | 1.364 | 1.144 | 0.381 | 0.000e+00 |
| 1024 | 1 | 3 | guided_1 | 0 | 3 | 1 | 1.5845 | 0.0000 | 1.5845 | 1.355 | 1.137 | 0.379 | 0.000e+00 |
| 1024 | 1 | 3 | runtime | 0 | 3 | 1 | 1.7072 | 0.0000 | 1.7072 | 1.258 | 1.055 | 0.352 | 0.000e+00 |
| 1024 | 1 | 3 | static | 0 | 3 | 4 | 1.3015 | 0.0000 | 1.3015 | 1.653 | 1.384 | 0.461 | 0.000e+00 |
| 1024 | 1 | 3 | static_1 | 0 | 3 | 1 | 2.0852 | 0.0000 | 2.0852 | 1.030 | 0.864 | 0.288 | 0.000e+00 |
| 1024 | 3 | 1 | static | 0 | 3 | 3 | 1.2952 | 0.0314 | 1.3290 | 1.658 | 1.356 | 0.452 | 0.000e+00 |
| 1024 | 1 | 4 | static | 1 | 4 | 1 | 1.7670 | 0.0000 | 1.7671 | 1.215 | 1.019 | 0.255 | 0.000e+00 |
| 1024 | 1 | 1 | static | 0 | 1 | 3 | 1.8015 | 0.0000 | 1.8015 | 1.192 | 1.000 | 1.000 | 0.000e+00 |
| 2048 | 1 | 1 | static | 0 | 1 | 1 | 14.6306 | 0.0000 | 14.6306 | 1.174 | 1.257 | 1.257 | 0.000e+00 |
| 2048 | 1 | 2 | static | 0 | 2 | 1 | 10.6504 | 0.0000 | 10.6504 | 1.613 | 1.726 | 0.863 | 0.000e+00 |
| 2048 | 2 | 1 | static | 0 | 2 | 1 | 10.8198 | 0.1505 | 10.9703 | 1.588 | 1.676 | 0.838 | 0.000e+00 |
| 2048 | 1 | 3 | static | 0 | 3 | 1 | 10.2075 | 0.0000 | 10.2075 | 1.683 | 1.801 | 0.600 | 0.000e+00 |
| 2048 | 3 | 1 | static | 0 | 3 | 1 | 9.9654 | 0.1980 | 10.1634 | 1.724 | 1.809 | 0.603 | 0.000e+00 |
| 2048 | 1 | 1 | static | 0 | 1 | 1 | 18.3850 | 0.0000 | 18.3850 | 0.934 | 1.000 | 1.000 | 0.000e+00 |
| 4096 | 1 | 3 | static | 0 | 3 | 1 | 81.5136 | 0.0000 | 81.5136 | 1.686 | 1.413 | 0.471 | 0.000e+00 |
| 4096 | 1 | 1 | static | 0 | 1 | 1 | 115.1840 | 0.0000 | 115.1840 | 1.193 | 1.000 | 1.000 | 0.000e+00 |
| 8192 | 1 | 3 | static | 0 | 3 | 1 | 722.2105 | 0.0000 | 722.2105 | 1.522 | 1.285 | 0.428 | 0.000e+00 |
| 8192 | 1 | 1 | static | 0 | 1 | 1 | 928.3189 | 0.0000 | 928.3189 | 1.184 | 1.000 | 1.000 | 0.000e+00 |

## Amdahl estimate

Serial fraction fitted as s = (C * T(C) - T(1)) / (T(1) * (C - 1)), where C = P x T and T is total time.
Negative s means the parallel total time beat a pure 1/C split (noise or a tiny problem).

- N=1024 1x2: s=0.485, communication share=0.000, Amdahl bound=1.35, measured speedup=1.35
- N=1024 1x3: s=0.584, communication share=0.000, Amdahl bound=1.38, measured speedup=1.38
- N=1024 2x1: s=0.525, communication share=0.015, Amdahl bound=1.31, measured speedup=1.31
- N=1024 3x1: s=0.607, communication share=0.024, Amdahl bound=1.36, measured speedup=1.36
- N=2048 1x2: s=0.159, communication share=0.000, Amdahl bound=1.73, measured speedup=1.73
- N=2048 1x3: s=0.333, communication share=0.000, Amdahl bound=1.80, measured speedup=1.80
- N=2048 2x1: s=0.193, communication share=0.014, Amdahl bound=1.68, measured speedup=1.68
- N=2048 3x1: s=0.329, communication share=0.019, Amdahl bound=1.81, measured speedup=1.81
- N=4096 1x3: s=0.562, communication share=0.000, Amdahl bound=1.41, measured speedup=1.41
- N=8192 1x3: s=0.667, communication share=0.000, Amdahl bound=1.29, measured speedup=1.29

Gustafson speedup is P - s(P-1). This study keeps N fixed, so it is strong scaling and Amdahl is the model that applies. Gustafson would require growing N with the core count.
