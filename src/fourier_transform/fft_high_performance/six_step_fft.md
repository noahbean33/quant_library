# Six-Step FFT Algorithm

## Algorithm

Computes a 1-D FFT of length `n = n1 * n2` by viewing the data as an `n1 × n2` matrix, performing smaller FFTs along each dimension with twiddle factor multiplication in between. The six steps consist of transpositions and FFTs designed to exploit cache locality.

### Pseudocode

```
function SixStepFFT(x, n1, n2):
    // x is an n1 × n2 complex matrix
    // y is an n2 × n1 complex matrix (workspace)
    // w is an n1 × n2 twiddle factor matrix:  w[i][j] = e^(-2πi·i·j / n)

    // Step 1: Transposition  (n1 × n2 → n2 × n1)
    for i = 0 to n1 - 1:
        for j = 0 to n2 - 1:
            y[j][i] = x[i][j]

    // Step 2: n1 individual n2-point FFTs (column-wise)
    for i = 0 to n1 - 1:
        FFT(y[0..n2-1][i], n2)

    // Steps 3–4: Twiddle factor multiplication + transposition (n2 × n1 → n1 × n2)
    for j = 0 to n2 - 1:
        for i = 0 to n1 - 1:
            x[i][j] = y[j][i] * w[i][j]

    // Step 5: n2 individual n1-point FFTs (column-wise)
    for j = 0 to n2 - 1:
        FFT(x[0..n1-1][j], n1)

    // Step 6: Transposition  (n1 × n2 → n2 × n1)
    for i = 0 to n1 - 1:
        for j = 0 to n2 - 1:
            y[j][i] = x[i][j]

    return y
```

### Notes

- The six-step FFT decomposes a length-`n` 1-D FFT into two sets of shorter FFTs using the **Cooley-Tukey factorization** `n = n1 × n2`.
- The transpositions convert column access into contiguous memory access, improving **cache performance**.
- The twiddle factors `w[i][j] = e^(-2πi·i·j/n)` account for the phase shift between the two dimensions of the decomposition.
- This is the foundation of many high-performance FFT implementations; the **blocked** variant further improves cache utilization.
