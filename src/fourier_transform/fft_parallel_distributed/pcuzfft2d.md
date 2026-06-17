# Parallel 2-D Complex FFT for NVIDIA GPUs (CUDA Fortran + MPI)

## Overview

MPI-parallel 2-D complex FFT using **NVIDIA GPUs** for local computation. Each MPI process offloads its local FFTs to a GPU via cuFFT, with MPI all-to-all for inter-process data redistribution.

### Interface

```
CALL PZFFT2D(A, B, NX, NY, ICOMM, NPU, IOPT)

A(NX, NY/NPU) — complex*16 input (block-distributed along Y)
B(...)         — complex*16 output
NX, NY         — transform lengths: (2^p)(3^q)(5^r)
ICOMM          — MPI communicator
NPU            — number of MPI processes
IOPT           — 0: create plan, -1/+1: forward/inverse (same dist),
                 -2/+2: forward/inverse (transposed dist), 3: destroy plan
```

### Algorithm

1. Copy local data to GPU
2. GPU: cuFFT NX-point batched FFTs along X
3. Copy to host → MPI_Alltoall → copy to GPU
4. GPU: cuFFT NY-point batched FFTs along Y
5. Copy result to host

### Notes

- GPU handles transpositions and FFTs; MPI handles inter-process communication
- `IOPT = ±2` avoids one all-to-all by using transposed distribution

## Source

Original Fortran: `pcuzfft2d.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
