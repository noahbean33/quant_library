# Multirow FFT Based on Stockham FFT Algorithm

## Algorithm

A vectorized variant of the Stockham FFT that processes `ns` independent rows simultaneously. Each row undergoes the same butterfly operations with the same twiddle factors, enabling SIMD/vector parallelism across rows.

Given `ns` rows of `n = 2^p` complex values each, compute the DFT of every row using the Stockham auto-sort approach with an additional inner loop over rows.

### Pseudocode

```
function MultirowStockhamFFT(X, ns, n):
    // X_prev, X_next: ns × n complex arrays (two buffers)
    // n = 2^p

    l = n / 2
    m = 1

    for t = 1 to p:
        ω = e^(-2πi / (2l))
        for j = 0 to l - 1:
            for k = 0 to m - 1:
                for row = 0 to ns - 1:         // vectorizable loop
                    c0 = X_prev[row][k + j*m]
                    c1 = X_prev[row][k + j*m + l*m]
                    X_next[row][k + 2*j*m]     = c0 + c1
                    X_next[row][k + 2*j*m + m] = ω^j * (c0 - c1)

        l = l / 2
        m = m * 2
        swap(X_prev, X_next)

    return X_prev
```

### Notes

- The innermost `row` loop is **independent across rows** — no data dependencies — making it ideal for vectorization (SIMD) or GPU warp-level parallelism.
- This is identical to the standard Stockham FFT except for the added `row` dimension; the twiddle factors `ω^j` are shared across all rows.
- `ns` is typically chosen to match the vector register width (e.g., 2 for SSE, 4 for AVX, 32 for GPU warps).
- The Stockham formulation naturally avoids bit-reversal permutation — output is in natural order.
- This is used in FFTE as the core vectorized kernel for high-performance multi-column FFTs.
