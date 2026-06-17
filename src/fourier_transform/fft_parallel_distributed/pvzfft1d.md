# Parallel 1-D Complex FFT for Vector Machines (MPI)

## Overview

MPI-parallel 1-D complex FFT optimized for **vector machines** (e.g., NEC SX, Fujitsu VPP). Same algorithm and interface as `PZFFT1D` but uses vector-optimized local FFT kernels (`VZFFT1D`) that exploit long vector lengths for high throughput on vector processors.

### Interface

```
CALL PZFFT1D(A, B, W, N, ICOMM, ME, NPU, IOPT)

A(N/NPU) — complex*16 input (block-distributed)
B(N/NPU) — complex*16 output (block-distributed)
W(N/NPU) — complex*16 coefficient vector
N        — transform length (integer*8): (2^p)(3^q)(5^r)
ICOMM    — MPI communicator
ME       — MPI rank
NPU      — number of MPI processes
IOPT     — 0: init, -1: forward, +1: inverse
```

### Notes

- Identical parallel decomposition to `pzfft1d.f` — six-step with all-to-all communication
- The local FFT kernels are replaced with vector-optimized versions that maximize vector pipeline utilization
- Uses `mpif.h` include (Fortran 77 MPI binding) rather than Fortran 90 `use mpi`
- Best performance on machines with long vector registers (256+ elements)

## Source

Original Fortran: `pvzfft1d.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
