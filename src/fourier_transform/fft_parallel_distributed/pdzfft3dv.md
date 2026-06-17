# Parallel 3-D Real-to-Complex FFT (MPI, 2-D Decomposition)

## Overview

MPI-parallel 3-D real-to-complex FFT with **2-D domain decomposition** for an `NX × NY × NZ` real array. Data is distributed across a `NPUY × NPUZ` process grid, with each process holding `NX × (NY/NPUY) × (NZ/NPUZ)` real elements.

### Interface

```
CALL PDZFFT3DV(A, B, NX, NY, NZ, ICOMMY, ICOMMZ, MEY, NPUY, NPUZ, IOPT)

A(NX, NY/NPUY, NZ/NPUZ)       — real*8 input / complex*16 output (in-place)
B(NX/2+1, NY/NPUY, NZ/NPUZ)   — complex*16 work vector
NX, NY, NZ                     — transform lengths: (2^p)(3^q)(5^r)
ICOMMY, ICOMMZ                 — MPI communicators for Y and Z directions
MEY                            — rank in Y-direction
NPUY, NPUZ                     — process counts in Y and Z directions
IOPT                           — 0: init, -1: forward (same dist), -2: forward (transposed)
```

### Notes

- Combines real-to-complex FFT with 2-D decomposition for maximum scalability
- Two all-to-all communications (one per decomposition dimension)
- Output exploits Hermitian symmetry: only `NX/2+1` frequencies stored in X
- `IOPT = -2` leaves output in `(BLOCK,BLOCK,*)` distribution

## Source

Original Fortran: `pdzfft3dv.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
