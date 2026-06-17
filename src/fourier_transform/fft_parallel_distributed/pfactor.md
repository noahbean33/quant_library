# Parallel Factorization Routines

## Overview

Utility routines for factoring FFT lengths into `2^p × 3^q × 5^r` components and choosing optimal `NX × NY` decompositions for the parallel six-step FFT.

### FACTOR8

Factors a 64-bit integer `N` into powers of 2, 3, and 5: `N = 2^IP(1) × 3^IP(2) × 5^IP(3)`.

### PGETNXNY

Chooses `NX` and `NY` such that `N = NX × NY`, with `NX ≈ √N`, and both `NX` and `NY` are divisible by `NPU` (number of MPI processes). The factorization ensures `NX` and `NY` are also of the form `2^p × 3^q × 5^r`.

### Pseudocode

```
function FACTOR8(N):
    IP = [0, 0, 0]
    N2 = N
    while N2 > 1:
        if N2 % 2 == 0: IP[0]++; N2 /= 2
        else if N2 % 3 == 0: IP[1]++; N2 /= 3
        else if N2 % 5 == 0: IP[2]++; N2 /= 5
        else: break
    return IP

function PGETNXNY(N, NPU):
    sqrtN = floor(√N)
    factor NPU into LNPU = [p2, p3, p5]
    factor N into IP = [p2, p3, p5]

    // Search for NX closest to √N, with NX divisible by NPU
    best_NX = 1
    for k = LNPU[2] to (IP[2]+1)/2:
        for j = LNPU[1] to (IP[1]+1)/2:
            for i = LNPU[0] to (IP[0]+1)/2:
                NX = 2^i × 3^j × 5^k
                if NX ≤ sqrtN and (sqrtN - NX) < best_residual:
                    record NX factors

    NY = N / NX
    return (NX, NY)
```

### Notes

- The constraint `NX ≥ NPU` (ensured by starting the search at `LNPU[i]`) guarantees that both `NX` and `NY` are divisible by `NPU`.
- Choosing `NX ≈ √N` balances the work in the two FFT phases of the six-step algorithm.

## Source

Original Fortran: `pfactor.f` — FFTE package by Daisuke Takahashi, University of Tsukuba.
