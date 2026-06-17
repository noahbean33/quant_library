# 3-D Complex FFT for NVIDIA GPUs (CUDA Fortran)

## Overview

A 3-D complex FFT routine for NVIDIA GPUs using cuFFT. Computes the DFT of an `NX × NY × NZ` complex array where each dimension has length `(2^p)(3^q)(5^r)`. Uses transpositions to make each dimension contiguous before calling cuFFT batched 1-D FFTs.

### Interface

```
CALL ZFFT3D(A, NX, NY, NZ, IOPT)

A(NX,NY,NZ) — complex*16 input/output array
NX          — transform length in X: (2^p)(3^q)(5^r)
NY          — transform length in Y: (2^p)(3^q)(5^r)
NZ          — transform length in Z: (2^p)(3^q)(5^r)
IOPT        — 0: create plan, -1: forward, +1: inverse, 3: destroy plan
```

### Algorithm

```
function ZFFT3D(A, NX, NY, NZ, IOPT):
    if IOPT == 0:
        create cuFFT plans:
          PLANX: NX-point, batch NY*NZ
          PLANY: NY-point, batch NZ*NX
          PLANZ: NZ-point, batch NX*NY
        return
    if IOPT == 3:  destroy all plans; return

    copy A → device A_d

    if inverse (IOPT == 1):
        conjugate all elements

    // FFT along Z
    ZTRANS(A_d, B_d, NX*NY, NZ)         // make Z contiguous
    cuFFT NZ-point batched FFT on B_d

    // FFT along Y
    ZTRANS(B_d, A_d, NZ*NX, NY)         // make Y contiguous
    cuFFT NY-point batched FFT on A_d

    // FFT along X
    ZTRANS(A_d, B_d, NY*NZ, NX)         // make X contiguous
    cuFFT NX-point batched FFT: B_d → A_d

    if inverse:
        conjugate and scale by 1/(NX*NY*NZ)

    copy A_d → A
```

### Notes

- Three transpositions cycle through the dimensions, making each one contiguous in turn for the batched cuFFT call.
- The transpose sizes are `NX*NY × NZ`, `NZ*NX × NY`, and `NY*NZ × NX` — each "flattens" two dimensions and swaps with the target dimension.
- No twiddle factors between dimensions (true 3-D DFT, not a 1-D decomposition).
- The final cuFFT writes directly from `B_d` to `A_d` (`cufftExecZ2Z(PLANX, B_d, A_d, ...)`).
- Three cuFFT plans are created, one per dimension, each with the appropriate batch count.

## Source

Original Fortran: `cuzfft3d.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
