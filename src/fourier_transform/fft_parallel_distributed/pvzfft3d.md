# Parallel 3-D Complex FFT for Vector Machines (MPI)

## Overview

MPI-parallel 3-D complex FFT optimized for **vector machines**. Same algorithm and interface as `PZFFT3D` (1-D slab decomposition along Z) but uses vector-optimized local FFT kernels.

### Interface

```
CALL PZFFT3D(A, B, NX, NY, NZ, ICOMM, NPU, IOPT)

A(NX, NY, NZ/NPU) — complex*16 input (block-distributed along Z)
B(...)             — complex*16 output
NX, NY, NZ         — transform lengths: (2^p)(3^q)(5^r)
ICOMM              — MPI communicator
NPU                — number of MPI processes
IOPT               — 0: init, -1/+1: forward/inverse (same dist),
                     -2/+2: forward/inverse (transposed dist)
```

### Notes

- Same parallel decomposition as `pzfft3d.f` with vector-optimized local kernels
- X and Y FFTs are fully local; one all-to-all redistributes for Z FFTs
- `IOPT = ±2` avoids final redistribution, leaving output in `(BLOCK,*,*)` distribution

## Source

Original Fortran: `pvzfft3d.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
