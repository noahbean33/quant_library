# Parallel 3-D Complex-to-Real FFT (MPI, 2-D Decomposition)

## Overview

MPI-parallel 3-D complex-to-real (inverse) FFT with **2-D domain decomposition** for an `NX × NY × NZ` real output. Data is distributed across a `NPUY × NPUZ` process grid. The inverse of `PDZFFT3DV`.

### Interface

```
CALL PZDFFT3DV(A, B, NX, NY, NZ, ICOMMY, ICOMMZ, MEY, NPUY, NPUZ, IOPT)

A(NX/2+1, NY/NPUY, NZ/NPUZ) — complex*16 input / real*8 output (in-place)
B(NX/2+1, NY/NPUY, NZ/NPUZ) — complex*16 work vector
NX, NY, NZ                   — transform lengths: (2^p)(3^q)(5^r)
ICOMMY, ICOMMZ               — MPI communicators for Y and Z directions
MEY                          — rank in Y-direction
NPUY, NPUZ                   — process counts in Y and Z directions
IOPT                         — 0: init, +1: inverse (same dist), +2: inverse (transposed input)
```

### Algorithm

1. All-to-all along Z communicator to make Z local
2. Local inverse complex FFTs along Z
3. All-to-all along Y communicator to make Y local
4. Local inverse complex FFTs along Y
5. Local complex-to-real inverse FFTs along X
6. Output real data in `A` (in-place)

### Notes

- Inverse of `PDZFFT3DV`
- Two all-to-all communications (one per decomposition dimension)
- In-place: real output overwrites complex input in `A`

## Source

Original Fortran: `pzdfft3dv.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
