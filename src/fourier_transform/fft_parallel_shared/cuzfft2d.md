# 2-D Complex FFT for NVIDIA GPUs (CUDA Fortran)

## Overview

A 2-D complex FFT routine for NVIDIA GPUs using cuFFT. Computes the DFT of an `NX × NY` complex array where each dimension has length `(2^p)(3^q)(5^r)`. Uses transpositions to convert between row and column access patterns, with cuFFT batched 1-D FFTs along each dimension.

### Interface

```
CALL ZFFT2D(A, NX, NY, IOPT)

A(NX,NY) — complex*16 input/output array
NX       — transform length in X: (2^p)(3^q)(5^r)
NY       — transform length in Y: (2^p)(3^q)(5^r)
IOPT     — 0: create plan, -1: forward, +1: inverse, 3: destroy plan
```

### Algorithm

```
function ZFFT2D(A, NX, NY, IOPT):
    if IOPT == 0:  create cuFFT plans for NX (batch NY) and NY (batch NX); return
    if IOPT == 3:  destroy plans; return

    copy A → device A_d

    if inverse (IOPT == 1):
        conjugate all elements

    ZTRANS(A_d, B_d, NX, NY)             // transpose NX×NY → NY×NX
    cuFFT NY-point batched FFT on B_d    // NY FFTs of length NY (along Y)
    ZTRANS(B_d, A_d, NY, NX)             // transpose NY×NX → NX×NY
    cuFFT NX-point batched FFT on A_d    // NX FFTs of length NX (along X)

    if inverse:
        conjugate and scale by 1/(NX*NY)

    copy A_d → A
```

### Notes

- The 2-D FFT is decomposed into two batched 1-D FFTs with transpositions, exploiting cuFFT's efficient batched execution.
- No twiddle factors are needed between dimensions (unlike the 1-D six-step FFT) because this is a true 2-D DFT, not a 1-D decomposition.
- Inverse transform uses the conjugate-forward-conjugate-scale trick.
- Plans are created once and reused.

## Source

Original Fortran: `cuzfft2d.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
