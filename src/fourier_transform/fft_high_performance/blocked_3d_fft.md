# Blocked Three-Dimensional FFT Algorithm

## Algorithm

A cache-optimized 3-D FFT that applies the row-column approach across all three dimensions with blocked transpositions. FFTs are performed along dimensions 1, 2, and 3 in sequence, using workspace buffers to ensure contiguous memory access.

### Pseudocode

```
function Blocked3D_FFT(x, n1, n2, n3, nb):
    // x: n1 × n2 × n3 complex array
    // nb: cache block size
    // ywork: (n2+np) × nb workspace for dim-2 FFTs
    // zwork: (n3+np) × nb workspace for dim-3 FFTs

    // Step 1: n2*n3 individual n1-point column FFTs
    for k = 0 to n3 - 1:
        for j = 0 to n2 - 1:
            FFT(x[0..n1-1][j][k], n1)

        // Steps 2-4: Blocked dim-2 FFTs with transposition
        for ii = 0 to n1-1 step nb:
            ib = min(nb, n1 - ii)

            // Step 2: Blocked transposition into ywork
            for i = ii to ii + ib - 1:
                for j = 0 to n2 - 1:
                    ywork[j][i - ii] = x[i][j][k]

            // Step 3: n1*n3 individual n2-point FFTs
            for i = 0 to ib - 1:
                FFT(ywork[0..n2-1][i], n2)

            // Step 4: Blocked transposition back
            for j = 0 to n2 - 1:
                for i = ii to ii + ib - 1:
                    x[i][j][k] = ywork[j][i - ii]

    // Steps 5-6: Blocked dim-3 FFTs with transposition
    for j = 0 to n2 - 1:
        for ii = 0 to n1-1 step nb:
            ib = min(nb, n1 - ii)

            // Blocked transposition into zwork
            for i = ii to ii + ib - 1:
                for k = 0 to n3 - 1:
                    zwork[k][i - ii] = x[i][j][k]

            // Step 5: n1*n2 individual n3-point FFTs
            for i = 0 to ib - 1:
                FFT(zwork[0..n3-1][i], n3)

            // Step 6: Blocked transposition back
            for k = 0 to n3 - 1:
                for i = ii to ii + ib - 1:
                    x[i][j][k] = zwork[k][i - ii]
```

### Notes

- Unlike the nine-step algorithm, this variant does **not** use twiddle factors between dimensions — it is a pure row-column 3-D FFT.
- The blocked transpositions into `ywork` and `zwork` convert strided dimension-2 and dimension-3 accesses into contiguous memory operations.
- Dimension-1 FFTs are naturally contiguous and need no transposition.
- The padding `np` in the workspace arrays prevents cache-line conflicts for power-of-two sizes.
