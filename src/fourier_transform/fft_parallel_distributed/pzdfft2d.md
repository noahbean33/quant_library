# Parallel 2-D Complex-to-Real FFT (MPI)

## Overview

MPI-parallel 2-D complex-to-real (inverse) FFT for an `NX × NY` real output, block-distributed along Y across `NPU` processes. The inverse of `PDZFFT2D` — takes Hermitian-symmetric complex input `(NX/2+1) × NY` and produces real output `NX × NY`.

### Interface

```
CALL PZDFFT2D(A, B, NX, NY, ICOMM, ME, NPU, IOPT)

A(NX/2+1, NY/NPU)   — complex*16 input (IOPT=+1)
  or A((NX/2)/NPU+δ, NY) — complex*16 input (IOPT=+2, transposed)
B(NX, NY/NPU)        — real*8 output (block-distributed along Y)
NX, NY                — transform lengths: (2^p)(3^q)(5^r)
ICOMM                 — MPI communicator
ME                    — MPI rank
NPU                   — number of MPI processes
IOPT                  — 0: init, +1: inverse (same dist), +2: inverse (transposed input)
```

### Algorithm

1. Local complex inverse FFTs along Y
2. All-to-all to redistribute X-frequencies
3. Local complex-to-real inverse FFTs along X (exploit Hermitian symmetry)
4. Output real data

### Notes

- Inverse of `PDZFFT2D`: `IOPT=+1` matches `PDZFFT2D IOPT=-1`, `IOPT=+2` matches `IOPT=-2`
- The complex-to-real step recovers the full `NX` real values from `NX/2+1` complex frequencies

## Source

Original Fortran: `pzdfft2d.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
