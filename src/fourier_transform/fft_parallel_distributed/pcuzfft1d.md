# Parallel 1-D Complex FFT for NVIDIA GPUs (CUDA Fortran + MPI)

## Overview

MPI-parallel 1-D complex FFT using **NVIDIA GPUs** for local computation. Combines the six-step parallel decomposition (MPI all-to-all) with cuFFT for the local FFT phases. Each MPI process offloads its local FFTs to a GPU.

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
IOPT     — 0: create plan, -1: forward, +1: inverse, 3: destroy plan
```

### Algorithm

1. Copy local data to GPU
2. GPU: transposition + cuFFT batched NY-point FFTs
3. GPU: twiddle factor multiplication + transposition
4. Copy back to host for MPI communication
5. `MPI_Alltoall`
6. Copy to GPU
7. GPU: cuFFT batched NX-point FFTs + transposition
8. Copy result back to host

### Notes

- Each MPI process uses one GPU (typically via `cudaSetDevice(rank % num_gpus)`)
- Host↔GPU data transfers occur around the all-to-all communication (which is host-based MPI)
- cuFFT plans are created with `IOPT=0` and destroyed with `IOPT=3`
- GPU kernels handle transpositions and twiddle factor multiplication

## Source

Original Fortran: `pcuzfft1d.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
