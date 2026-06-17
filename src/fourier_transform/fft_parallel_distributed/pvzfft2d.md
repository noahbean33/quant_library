# Parallel 2-D Complex FFT for Vector Machines (MPI)

## Overview

MPI-parallel 2-D complex FFT optimized for **vector machines**. Same algorithm and interface as `PZFFT2D` but uses vector-optimized local FFT kernels for high throughput on vector processors.

### Interface

```
CALL PZFFT2D(A, B, NX, NY, ICOMM, NPU, IOPT)

A(NX, NY/NPU) — complex*16 input (block-distributed along Y)
B(...)         — complex*16 output
NX, NY         — transform lengths: (2^p)(3^q)(5^r)
ICOMM          — MPI communicator
NPU            — number of MPI processes
IOPT           — 0: init, -1/+1: forward/inverse (same dist),
                 -2/+2: forward/inverse (transposed dist)
```

### Notes

- Same parallel decomposition as `pzfft2d.f` with vector-optimized local kernels
- `IOPT = ±2` variants avoid one all-to-all by leaving output in transposed distribution

## Source

Original Fortran: `pvzfft2d.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
