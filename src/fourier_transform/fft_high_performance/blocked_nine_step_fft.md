# Blocked Nine-Step FFT Algorithm

## Algorithm

A cache-optimized 3-D FFT algorithm that decomposes a length `n = n1 × n2 × n3` transform into three sets of 1-D FFTs with blocked transpositions between them. This is the 3-D generalization of the blocked six-step FFT.

### Pseudocode

```
function BlockedNineStepFFT(x, y, u2, u3, n1, n2, n3, nb):
    // x: n1 × n2 × n3 input (row-major)
    // y: n3 × n2 × n1 output
    // u2: n3 × n2 twiddle factors (for step 3)
    // u3: n1 × n2 × n3 twiddle factors (for step 6)
    // nb: cache block size

    // --- Phase 1: FFTs along dimension 3, with blocked transposition ---
    for j = 0 to n2 - 1:
        for ii = 0 to n1-1 step nb:
            ib = min(nb, n1 - ii)

            // Step 1: Blocked transposition into zwork
            for kk = 0 to n3-1 step nb:
                kb = min(nb, n3 - kk)
                for i = ii to ii + ib - 1:
                    for k = kk to kk + kb - 1:
                        zwork[k][i - ii] = x[i][j][k]

            // Step 2: n1*n2 individual n3-point FFTs
            for i = ii to ii + ib - 1:
                FFT(zwork[0..n3-1][i - ii], n3)

            // Steps 3-4: Twiddle multiplication + transposition back
            for k = 0 to n3 - 1:
                for i = ii to ii + ib - 1:
                    x[i][j][k] = zwork[k][i - ii] * u2[k][j]

    // --- Phase 2: FFTs along dimension 2, with blocked transposition ---
    for k = 0 to n3 - 1:
        for ii = 0 to n1-1 step nb:
            ib = min(nb, n1 - ii)

            // Blocked transposition into ywork
            for jj = 0 to n2-1 step nb:
                jb = min(nb, n2 - jj)
                for i = ii to ii + ib - 1:
                    for j = jj to jj + jb - 1:
                        ywork[j][i - ii] = x[i][j][k]

            // Step 5: n1*n3 individual n2-point FFTs
            for i = ii to ii + ib - 1:
                FFT(ywork[0..n2-1][i - ii], n2)

            // Steps 6-7: Twiddle multiplication + transposition back
            for j = 0 to n2 - 1:
                for i = ii to ii + ib - 1:
                    x[i][j][k] = ywork[j][i - ii] * u3[i][j][k]

        // Step 8: n2*n3 individual n1-point FFTs
        for j = 0 to n2 - 1:
            FFT(x[0..n1-1][j][k], n1)

    // Step 9: Blocked transposition (n1 × n2 × n3 → n3 × n2 × n1)
    for ii = 0 to n1-1 step nb:
        for jj = 0 to n2-1 step nb:
            for kk = 0 to n3-1 step nb:
                for i, j, k in block:
                    y[k][j][i] = x[i][j][k]
```

### Notes

- This algorithm processes the 3-D array one dimension at a time, using **blocked transpositions** to ensure data locality.
- Two workspace arrays (`ywork`, `zwork`) of size `(n2+np) × nb` and `(n3+np) × nb` hold blocks of columns during the FFT and twiddle steps.
- The twiddle factors `u2[k][j]` and `u3[i][j][k]` account for the phase shifts between the three dimensions.
- The final transposition (step 9) reorders the output as `y(k, j, i)` — a full 3-D transpose.
- This is the 3-D extension of Takahashi's blocking strategy, achieving near-optimal cache reuse for large 3-D FFTs.
