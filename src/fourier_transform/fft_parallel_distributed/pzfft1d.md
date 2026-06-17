# Parallel 1-D Complex FFT (MPI)

## Overview

MPI-parallel 1-D complex FFT for lengths `N = (2^p)(3^q)(5^r)`. Uses the six-step algorithm with block distribution across `NPU` MPI processes. Each process holds `N/NPU` elements.

### Interface

```
CALL PZFFT1D(A, B, W, N, ICOMM, ME, NPU, IOPT)

A(N/NPU)   — complex*16 input vector (block-distributed)
B(N/NPU)   — complex*16 output vector (block-distributed)
W(N/NPU)   — complex*16 coefficient/work vector
N          — transform length (integer*8): (2^p)(3^q)(5^r)
ICOMM      — MPI communicator
ME         — MPI rank
NPU        — number of MPI processes
IOPT       — 0: initialize coefficients, -1: forward, +1: inverse
```

### Algorithm

1. Factor `N` into `NX × NY` with `NX ≈ √N`, both divisible by `NPU`
2. Perform local transpositions and rearrangements
3. `NX/NPU` individual `NY`-point multicolumn FFTs (local)
4. Twiddle factor multiplication
5. All-to-all communication (`MPI_Alltoall`)
6. `NY/NPU` individual `NX`-point multicolumn FFTs (local)
7. Final rearrangement and transposition

### Notes

- Uses OpenMP for shared-memory parallelism within each MPI process
- Coefficients (twiddle factors) are precomputed with `IOPT=0` and stored in `W`
- The local FFTs call the FFTE mixed-radix kernels (`fft235`, `kernel`)
- Inverse transform conjugates input, performs forward FFT, conjugates and scales output

## Source

Original Fortran: `pzfft1d.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
