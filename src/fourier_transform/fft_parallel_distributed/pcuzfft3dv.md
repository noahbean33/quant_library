# Parallel 3-D Complex FFT for NVIDIA GPUs (CUDA Fortran + MPI, 2-D Decomposition)

## Overview

MPI-parallel 3-D complex FFT using **NVIDIA GPUs** with **2-D domain decomposition**. Data is distributed across a `NPUY × NPUZ` process grid, with each process holding `NX × (NY/NPUY) × (NZ/NPUZ)` elements. Combines cuFFT for local GPU computation with MPI all-to-all across two communicators.

### Interface

```
CALL PZFFT3DV(A, B, NX, NY, NZ, ICOMMY, ICOMMZ, NPUY, NPUZ, IOPT)

A(NX, NY/NPUY, NZ/NPUZ) — complex*16 input
B(...)                    — complex*16 output
NX, NY, NZ                — transform lengths: (2^p)(3^q)(5^r)
ICOMMY, ICOMMZ            — MPI communicators for Y and Z directions
NPUY, NPUZ                — process counts in Y and Z directions
IOPT                      — 0: create plan, -1/+1: forward/inverse (same dist),
                            -2/+2: forward/inverse (transposed dist), 3: destroy plan
```

### Algorithm

1. GPU: cuFFT along X (fully local)
2. GPU→Host, MPI_Alltoall along Y comm, Host→GPU
3. GPU: cuFFT along Y
4. GPU→Host, MPI_Alltoall along Z comm, Host→GPU
5. GPU: cuFFT along Z
6. Optional: redistribution for output layout

### Notes

- 2-D decomposition enables scaling to `NPUY × NPUZ` total processes
- Two all-to-all communications, each involving only a subset of processes
- GPU handles all transpositions and FFTs for maximum throughput
- `IOPT = ±2` avoids final redistribution

## Source

Original Fortran: `pcuzfft3dv.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
