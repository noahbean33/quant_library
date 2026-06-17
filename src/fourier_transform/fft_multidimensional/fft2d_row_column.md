# Two-Dimensional FFT Based on Row-Column Algorithm

## Algorithm

Computes a 2-D DFT of an `n1 × n2` complex array using the **row-column** (or **slab**) decomposition. Each dimension is transformed independently using 1-D FFTs, with a transposition between the two passes to maintain column-wise access.

### Pseudocode

```
function FFT2D(x, n1, n2):
    // x is an n1 × n2 complex matrix

    // Step 1: n2 individual n1-point column FFTs
    for j = 0 to n2 - 1:
        FFT(x[0..n1-1][j], n1)

    // Step 2: Transposition (n1 × n2 → n2 × n1)
    for i = 0 to n1 - 1:
        for j = 0 to n2 - 1:
            y[j][i] = x[i][j]

    // Step 3: n1 individual n2-point column FFTs
    for i = 0 to n1 - 1:
        FFT(y[0..n2-1][i], n2)

    // Step 4: Transposition (n2 × n1 → n1 × n2)
    for j = 0 to n2 - 1:
        for i = 0 to n1 - 1:
            x[i][j] = y[j][i]

    return x
```

### Notes

- The row-column algorithm is the simplest approach to multidimensional FFTs — it applies 1-D FFTs along each dimension independently.
- The **transposition** steps convert strided column access into contiguous memory access, which improves cache performance.
- Total work: `n2` FFTs of length `n1` plus `n1` FFTs of length `n2`, giving `O(n1·n2·(log n1 + log n2))` complexity.
- This is the foundation for more advanced 2-D FFT algorithms (e.g., the six-step FFT, blocked variants).
