# Parallel 3-D Real-to-Complex FFT (MPI, 1-D Decomposition)

## Overview

MPI-parallel 3-D real-to-complex FFT for an `NX × NY × NZ` real array, block-distributed along Z across `NPU` processes. Exploits Hermitian symmetry in X: output has dimensions `(NX/2+1) × NY × NZ` complex values.

### Interface

```
CALL PDZFFT3D(A, B, NX, NY, NZ, ICOMM, ME, NPU, IOPT)

A(NX, NY, NZ/NPU)       — real*8 input / complex*16 output (in-place)
B(NX/2+1, NY, NZ/NPU)   — complex*16 work vector
NX, NY, NZ               — transform lengths: (2^p)(3^q)(5^r)
ICOMM                    — MPI communicator
ME                       — MPI rank
NPU                      — number of MPI processes
IOPT                     — 0: init, -1: forward (same dist), -2: forward (transposed)
```

### Algorithm

1. Local real-to-complex FFTs along X (Hermitian symmetry → half output)
2. Local complex FFTs along Y
3. Rearrangement + all-to-all to redistribute Z
4. Local complex FFTs along Z
5. Output rearrangement

### Notes

- Output overwrites input array `A` (in-place), `B` is workspace
- `IOPT = -2` leaves output in `(BLOCK,*,*)` distribution
- Rank 0 may hold one extra X-frequency element (DC/Nyquist)

## Source

Original Fortran: `pdzfft3d.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
