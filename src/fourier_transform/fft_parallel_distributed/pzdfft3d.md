# Parallel 3-D Complex-to-Real FFT (MPI, 1-D Decomposition)

## Overview

MPI-parallel 3-D complex-to-real (inverse) FFT for an `NX × NY × NZ` real output, block-distributed along Z across `NPU` processes. The inverse of `PDZFFT3D` — takes Hermitian-symmetric complex input `(NX/2+1) × NY × NZ` and produces real output.

### Interface

```
CALL PZDFFT3D(A, B, NX, NY, NZ, ICOMM, ME, NPU, IOPT)

A(NX/2+1, NY, NZ/NPU) — complex*16 input / real*8 output (in-place)
B(NX/2+1, NY, NZ/NPU) — complex*16 work vector
NX, NY, NZ             — transform lengths: (2^p)(3^q)(5^r)
ICOMM                  — MPI communicator
ME                     — MPI rank
NPU                    — number of MPI processes
IOPT                   — 0: init, +1: inverse (same dist), +2: inverse (transposed input)
```

### Algorithm

1. All-to-all to redistribute Z → make Z local
2. Local inverse complex FFTs along Z
3. Local inverse complex FFTs along Y
4. Local complex-to-real inverse FFTs along X (exploit Hermitian symmetry)
5. Output real data in `A` (in-place)

### Notes

- Inverse of `PDZFFT3D`: `IOPT=+1` matches `-1`, `IOPT=+2` matches `-2`
- In-place: output real data overwrites input array `A`
- `B` is workspace for the communication/transposition phases

## Source

Original Fortran: `pzdfft3d.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
