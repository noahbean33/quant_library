# 1-D Complex FFT for NVIDIA GPUs (CUDA Fortran)

## Overview

A 1-D complex FFT routine for NVIDIA GPUs using cuFFT. Decomposes a length-`N` transform (where `N = (2^p)(3^q)(5^r)`) into two shorter FFTs using the six-step approach: `N = NX × NY`. The data is transposed between GPU FFT calls, with twiddle factor multiplication applied between the two dimensions.

### Interface

```
CALL ZFFT1D(A, N, IOPT, B)

A(N)   — complex*16 input/output array
B(2*N) — complex*16 work/coefficient array
N      — transform length: (2^p)(3^q)(5^r)
IOPT   — 0: create plan, -1: forward, +1: inverse, 3: destroy plan
```

### Algorithm (Six-Step on GPU)

```
function ZFFT1D(A, N, IOPT, B):
    factor N into NX × NY

    if IOPT == 0:  create cuFFT plans for NX and NY; return
    if IOPT == 3:  destroy cuFFT plans; return

    copy A → device A_d

    if inverse (IOPT == 1):
        conjugate all elements of A_d

    // Six-step FFT on GPU:
    ZTRANS(A_d, B_d, NX, NY)           // Step 1: transpose
    cuFFT NY-point batched FFT on B_d  // Step 2: NY FFTs
    ZTRANSMUL(B_d, A_d, NY, NX)        // Steps 3-4: transpose + twiddle
    cuFFT NX-point batched FFT on A_d  // Step 5: NX FFTs
    ZTRANS(A_d, B_d, NX, NY)           // Step 6: transpose

    if inverse:
        conjugate and scale A_d by 1/N

    copy B_d → A
```

### ZTRANSMUL Subroutine

Combines blocked transposition with twiddle factor multiplication:
```
B_d[j, i] = A_d[i, j] × e^(-2πi·(i-1)·(j-1)/(NX·NY))
```
Uses `NB × NB` tile blocking with shared memory for coalesced access.

### Notes

- The six-step decomposition allows using cuFFT's batched 1-D FFT for both dimensions, which is highly efficient on GPUs.
- Inverse transform is computed via: conjugate → forward FFT → conjugate → scale by `1/N`.
- `ZTRANSMUL` fuses the twiddle multiplication with the transposition to halve the number of global memory passes.
- Plans are created once (`IOPT=0`) and reused for multiple transforms, amortizing cuFFT setup cost.

## Source

Original Fortran: `cuzfft1d.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
