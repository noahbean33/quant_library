# Parallel 3-D Complex FFT for NVIDIA GPUs (CUDA Fortran + MPI, 1-D Decomposition)

## Overview

MPI-parallel 3-D complex FFT using **NVIDIA GPUs** with 1-D slab decomposition along Z. Each MPI process offloads local FFTs along X, Y, and Z to a GPU via cuFFT. MPI all-to-all redistributes data for the Z-dimension FFTs.

### Interface

```
CALL PZFFT3D(A, B, NX, NY, NZ, ICOMM, NPU, IOPT)

A(NX, NY, NZ/NPU) — complex*16 input (block-distributed along Z)
B(...)             — complex*16 output
NX, NY, NZ         — transform lengths: (2^p)(3^q)(5^r)
ICOMM              — MPI communicator
NPU                — number of MPI processes
IOPT               — 0: create plan, -1/+1: forward/inverse (same dist),
                     -2/+2: forward/inverse (transposed dist), 3: destroy plan
```

### Algorithm

1. GPU: cuFFT along X and Y (fully local)
2. GPU→Host transfer, MPI_Alltoall, Host→GPU transfer
3. GPU: cuFFT along Z
4. Optional: second all-to-all for output redistribution

### Notes

- Three cuFFT plans (one per dimension) with appropriate batch counts
- GPU transpositions between dimensions for coalesced memory access
- `IOPT = ±2` avoids final redistribution

## Source

Original Fortran: `pcuzfft3d.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
