# Parallel 2-D Complex FFT (MPI)

## Overview

MPI-parallel 2-D complex FFT for an `NX × NY` array where each dimension has length `(2^p)(3^q)(5^r)`. Block-distributed along the Y-dimension across `NPU` processes. Each process holds `NX × (NY/NPU)` elements.

### Interface

```
CALL PZFFT2D(A, B, NX, NY, ICOMM, NPU, IOPT)

A(NX, NY/NPU) — complex*16 input (block-distributed along Y)
B(...)         — complex*16 output (distribution depends on IOPT)
NX, NY         — transform lengths: (2^p)(3^q)(5^r)
ICOMM          — MPI communicator
NPU            — number of MPI processes
IOPT           — 0: init, -1: forward (same dist), +1: inverse (same dist),
                 -2: forward (transposed output), +2: inverse (transposed input)
```

### Algorithm

1. Local `NY/NPU` individual `NX`-point FFTs along X (fully local)
2. Local transposition + pack for all-to-all
3. `MPI_Alltoall` — redistributes data so Y-dimension becomes local
4. Local rearrangement to assemble full Y-dimension
5. Local `NX/NPU` individual `NY`-point FFTs along Y
6. Final transposition / all-to-all for output distribution

### Notes

- `IOPT = ±1`: input and output have the same distribution `(*,BLOCK)`
- `IOPT = -2`: output is transposed to `(BLOCK,*)` distribution — avoids the final all-to-all, saving one communication round
- `IOPT = +2`: input is `(BLOCK,*)`, output is `(*,BLOCK)` — inverse of `-2`
- Uses blocked transpositions with workspace arrays for cache efficiency

## Source

Original Fortran: `pzfft2d.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
