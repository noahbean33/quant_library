# Blocked Two-Dimensional FFT Based on Row-Column Algorithm

## Algorithm

A cache-optimized 2-D FFT that computes column FFTs first, then uses a blocked transposition into a workspace buffer to perform row FFTs with good cache locality, and finally transposes back.

### Pseudocode

```
function Blocked2D_FFT(x, n1, n2, nb):
    // x: n1 × n2 complex matrix (column-major: x[i][j] = x[i + j*n1])
    // nb: cache block size

    // Step 1: n2 individual n1-point column FFTs
    for j = 0 to n2 - 1:
        FFT(x[0..n1-1][j], n1)

    for ii = 0 to n1-1 step nb:
        ib = min(nb, n1 - ii)

        // Step 2: Blocked transposition into work buffer
        for jj = 0 to n2-1 step nb:
            jb = min(nb, n2 - jj)
            for i = ii to ii + ib - 1:
                for j = jj to jj + jb - 1:
                    work[j][i - ii] = x[i][j]

        // Step 3: ib individual n2-point FFTs (on work columns)
        for i = 0 to ib - 1:
            FFT(work[0..n2-1][i], n2)

        // Step 4: Blocked transposition back
        for j = 0 to n2 - 1:
            for i = ii to ii + ib - 1:
                x[i][j] = work[j][i - ii]
```

### Notes

- The column FFTs (step 1) access contiguous memory and are naturally cache-friendly.
- The row FFTs (step 3) would have strided access in the original array, so data is first **transposed into a contiguous work buffer** in blocks of `nb` rows.
- The block size `nb` is tuned so that `nb` rows of length `n2` fit in L1/L2 cache.
- This is simpler than the six-step algorithm but achieves similar cache performance for 2-D FFTs.
