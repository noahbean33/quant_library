# Three-Dimensional FFT Based on Row-Column Algorithm

## Algorithm

Computes a 3-D DFT of an `n1 × n2 × n3` complex array using the **row-column** decomposition. Each of the three dimensions is transformed independently using 1-D FFTs, with transpositions between passes to maintain column-wise (contiguous) access.

### Pseudocode

```
function FFT3D(x, n1, n2, n3):
    // x is an n1 × n2 × n3 complex array

    // Step 1: n2·n3 individual n1-point column FFTs
    for k = 0 to n3 - 1:
        for j = 0 to n2 - 1:
            FFT(x[0..n1-1][j][k], n1)

    // Step 2: Transposition (n1 × n2 × n3 → n2 × n3 × n1)
    for all (i, j, k):
        y[j][k][i] = x[i][j][k]

    // Step 3: n3·n1 individual n2-point column FFTs
    for i = 0 to n1 - 1:
        for k = 0 to n3 - 1:
            FFT(y[0..n2-1][k][i], n2)

    // Step 4: Transposition (n2 × n3 × n1 → n3 × n1 × n2)
    for all (j, k, i):
        z[k][i][j] = y[j][k][i]

    // Step 5: n1·n2 individual n3-point column FFTs
    for j = 0 to n2 - 1:
        for i = 0 to n1 - 1:
            FFT(z[0..n3-1][i][j], n3)

    // Step 6: Transposition (n3 × n1 × n2 → n1 × n2 × n3)
    for all (k, i, j):
        x[i][j][k] = z[k][i][j]

    return x
```

### Notes

- The row-column algorithm extends naturally to three dimensions: apply 1-D FFTs along each axis, using transpositions to make the target axis contiguous.
- Six steps total: three FFT passes (one per dimension) interleaved with three transpositions.
- Total work: `O(n1·n2·n3·(log n1 + log n2 + log n3))`.
- The transpositions are the performance bottleneck for large arrays; **blocked** and **cache-oblivious** variants address this.
- This is the conceptual basis for the production FFTE 3-D routines (`ZFFT3D`), which add cache blocking and OpenMP parallelism.
