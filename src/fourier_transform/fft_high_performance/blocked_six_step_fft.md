# Blocked Six-Step FFT Algorithm

## Algorithm

A cache-optimized variant of the six-step FFT. The data (`n = n1 × n2`) is processed in blocks of size `nb` to keep working sets within cache. A workspace array `work` of size `(n2 + np) × nb` is used for blocked transposition and in-place column FFTs.

### Pseudocode

```
function BlockedSixStepFFT(x, y, w, n1, n2, nb):
    // x: n1 × n2 input/workspace
    // y: n2 × n1 output
    // w: n1 × n2 twiddle factors
    // nb: block size (tuned to cache)

    for ii = 0 to n1-1 step nb:
        ib = min(nb, n1 - ii)

        // Step 1: Blocked transposition into work buffer
        for jj = 0 to n2-1 step nb:
            jb = min(nb, n2 - jj)
            for i = ii to ii + ib - 1:
                for j = jj to jj + jb - 1:
                    work[j][i - ii] = x[i][j]

        // Step 2: n1 individual n2-point FFTs (on block of columns)
        for i = ii to ii + ib - 1:
            FFT(work[0..n2-1][i - ii], n2)

        // Steps 3–4: Twiddle factor multiplication + write-back transposition
        for j = 0 to n2 - 1:
            for i = ii to ii + ib - 1:
                x[i][j] = work[j][i - ii] * w[i][j]

    for jj = 0 to n2-1 step nb:
        jb = min(nb, n2 - jj)

        // Step 5: n2 individual n1-point FFTs (on block of columns)
        for j = jj to jj + jb - 1:
            FFT(x[0..n1-1][j], n1)

        // Step 6: Blocked transposition
        for i = 0 to n1 - 1:
            for j = jj to jj + jb - 1:
                y[j][i] = x[i][j]
```

### Notes

- The **block size `nb`** is chosen so that `nb` columns of length `n2` (or `n1`) fit in cache.
- The workspace `work` avoids strided memory access during the column FFTs — data is copied into contiguous memory first.
- `np` is padding added to `work` to avoid cache-line conflicts (power-of-two stride aliasing).
- This is the key optimization from Takahashi's "Blocking Algorithm for FFT on Cache-Based Processors" (2001).
