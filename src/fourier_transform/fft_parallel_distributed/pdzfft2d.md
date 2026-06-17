# Parallel 2-D Real-to-Complex FFT (MPI)

## Overview

MPI-parallel 2-D real-to-complex FFT for an `NX × NY` real array, block-distributed along Y across `NPU` processes. Exploits Hermitian symmetry: the output has dimensions `(NX/2+1) × NY` complex values (only non-redundant half).

### Interface

```
CALL PDZFFT2D(A, B, NX, NY, ICOMM, ME, NPU, IOPT)

A(NX, NY/NPU)       — real*8 input (block-distributed along Y)
B(NX/2+1, NY/NPU)   — complex*16 output (IOPT=-1)
  or B((NX/2)/NPU+δ, NY) — complex*16 output (IOPT=-2, transposed)
NX, NY               — transform lengths: (2^p)(3^q)(5^r)
ICOMM                — MPI communicator
ME                   — MPI rank
NPU                  — number of MPI processes
IOPT                 — 0: init, -1: forward (same dist), -2: forward (transposed)
```

### Algorithm

1. Local real-to-complex FFTs along X using Hermitian symmetry
2. Local transposition + pack
3. `MPI_Alltoall` to redistribute
4. Local complex FFTs along Y
5. Output rearrangement (distribution depends on `IOPT`)

### Notes

- The real-to-complex transform halves the output size in X due to conjugate symmetry: `X[k] = conj(X[N-k])`
- `IOPT = -2` leaves output in `(BLOCK,*)` distribution — saves one all-to-all
- The extra element on rank 0 (`NX/2/NPU + 1` vs `NX/2/NPU`) handles the DC and Nyquist components

## Source

Original Fortran: `pdzfft2d.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
