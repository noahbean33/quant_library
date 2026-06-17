# Parallel 3-D Complex FFT (MPI, 1-D Decomposition)

## Overview

MPI-parallel 3-D complex FFT for an `NX × NY × NZ` array, block-distributed along Z across `NPU` processes. Each process holds `NX × NY × (NZ/NPU)` elements. Uses all-to-all communication to redistribute data for the Z-dimension FFTs.

### Interface

```
CALL PZFFT3D(A, B, NX, NY, NZ, ICOMM, NPU, IOPT)

A(NX, NY, NZ/NPU) — complex*16 input (block-distributed along Z)
B(...)             — complex*16 output
NX, NY, NZ         — transform lengths: (2^p)(3^q)(5^r)
ICOMM              — MPI communicator
NPU                — number of MPI processes
IOPT               — 0: init, -1/+1: forward/inverse (same dist),
                     -2/+2: forward/inverse (transposed output/input)
```

### Algorithm

1. Local FFTs along X: `NY × (NZ/NPU)` individual `NX`-point FFTs
2. Local transposition (swap X and Y)
3. Local FFTs along Y: `NX × (NZ/NPU)` individual `NY`-point FFTs
4. Rearrangement + pack for all-to-all
5. `MPI_Alltoall` — redistributes data so Z becomes local
6. Local transposition to assemble full Z
7. Local FFTs along Z: `(NX/NPU) × NY` individual `NZ`-point FFTs
8. Final transposition + all-to-all for output distribution

### Notes

- Dimensions X and Y are fully local — their FFTs require no communication
- Only one or two all-to-all communications needed (depending on `IOPT`)
- `IOPT = ±2` skips the final redistribution, leaving output in transposed `(BLOCK,*,*)` distribution
- Uses blocked transpositions with workspace arrays for cache efficiency

## Source

Original Fortran: `pzfft3d.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
