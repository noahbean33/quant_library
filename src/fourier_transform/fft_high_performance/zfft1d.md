# 1-D Complex FFT Routine (with OpenMP and Cache Blocking)

## Algorithm

Computes a 1-D complex FFT of length `n = 2^p * 3^q * 5^r`. For small transforms that fit in L2 cache, a direct mixed-radix FFT is used. For large transforms, a **blocked six-step FFT** approach is employed with OpenMP parallelism.

### Interface

```
zfft1d(a, n, iopt, b)
```

- **a[n]**: complex input/output vector
- **b[2n]**: work/coefficient vector
- **n**: transform length (`n = 2^p * 3^q * 5^r`)
- **iopt**: `0` = initialize coefficients, `-1` = forward FFT, `+1` = inverse FFT

### Pseudocode

```
function ZFFT1D(a, n, iopt, b):
    if iopt == +1:
        conjugate all elements of a

    if n ≤ L2_CACHE_SIZE / 48:
        // Small transform: direct mixed-radix FFT
        if iopt == 0:
            SETTBL(b[n:], n)
            return
        FFT235(a, b, b[n:], n)
    else:
        // Large transform: blocked six-step FFT
        factor n into nx * ny  (nx ≈ √n)

        if iopt == 0:
            SETTBL(wx, nx)
            SETTBL(wy, ny)
            SETTBL2(b[n:], nx, ny)
            return

        // OpenMP parallel region:
        ZFFT1D0(a, b, wx, wy, b[n:], nx, ny)

    if iopt == +1:
        conjugate and scale a by 1/n
```

### Inner Blocked Computation (ZFFT1D0)

```
function ZFFT1D0(a, b, wx, wy, w, nx, ny):
    // Phase 1: Blocked column FFTs (ny-point) with transpose
    for ii = 0 to nx-1 step NBLK:
        // Blocked transpose: a[nx × ny] columns → local buffer
        for i in block [ii, ii+NBLK):
            extract column i of a (length ny)
            FFT235(column, wy, ny)
            store result into b

    // Phase 2: Twiddle multiply + row FFTs (nx-point) with transpose
    for jj = 0 to ny-1 step NBLK:
        for j in block [jj, jj+NBLK):
            extract row j of b (length nx)
            multiply by twiddle factors w[·][j]
            FFT235(row, wx, nx)
            transpose-store into output a[ny × nx]
```

### Notes

- **Cache blocking** (parameter `NBLK = 16`) ensures that data accessed in the inner loops fits in L2 cache, minimizing cache misses during the transpose operations.
- **Padding** (parameter `NP = 8`) is added to work arrays to avoid cache-line conflicts.
- For small `n` (fitting in `L2SIZE / 48` complex elements), the overhead of the six-step decomposition is avoided by using a direct `FFT235` call.
- OpenMP parallelism is applied at the block level — each block of `NBLK` columns/rows is processed independently.
- The inverse transform is computed via conjugation: `IFFT(x) = conj(FFT(conj(x))) / n`.
