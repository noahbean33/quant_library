# Parallel 3-D Complex FFT (MPI, 2-D Decomposition)

## Overview

MPI-parallel 3-D complex FFT with **2-D domain decomposition** for an `NX × NY × NZ` array. Data is distributed across a 2-D process grid `NPUY × NPUZ`, with each process holding `NX × (NY/NPUY) × (NZ/NPUZ)` elements. This enables scaling to more processes than a 1-D slab decomposition.

### Interface

```
CALL PZFFT3DV(A, B, NX, NY, NZ, ICOMMY, ICOMMZ, NPUY, NPUZ, IOPT)

A(NX, NY/NPUY, NZ/NPUZ) — complex*16 input
B(...)                    — complex*16 output
NX, NY, NZ                — transform lengths: (2^p)(3^q)(5^r)
ICOMMY                    — MPI communicator in Y-direction
ICOMMZ                    — MPI communicator in Z-direction
NPUY                      — number of processes in Y-direction
NPUZ                      — number of processes in Z-direction
IOPT                      — 0: init, -1/+1: forward/inverse (same dist),
                            -2/+2: forward/inverse (transposed dist)
```

### Algorithm

1. Local FFTs along X (fully local, `NX`-point)
2. All-to-all along Y communicator — redistribute Y
3. Local FFTs along Y (`NY`-point)
4. All-to-all along Z communicator — redistribute Z
5. Local FFTs along Z (`NZ`-point)
6. Final redistribution (if needed for output layout)

### Notes

- Two separate communicators (`ICOMMY`, `ICOMMZ`) handle the two decomposition dimensions independently
- The 2-D decomposition allows `P = NPUY × NPUZ` processes where neither `NPUY` nor `NPUZ` needs to divide `NZ` alone
- Two all-to-all communications are needed (one per decomposition dimension), but each involves only `NPUY` or `NPUZ` processes — smaller messages, less contention
- `IOPT = ±2` leaves output in `(BLOCK,BLOCK,*)` distribution

## Source

Original Fortran: `pzfft3dv.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
