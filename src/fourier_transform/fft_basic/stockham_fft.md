# Stockham FFT Algorithm

## Algorithm

Given an input array `x` of `n = 2^p` complex values, compute the DFT using the Stockham auto-sort approach. This algorithm alternates between two buffers, eliminating the need for a separate bit-reversal permutation step.

### Pseudocode

```
function StockhamFFT(x, n):
    // n = 2^p
    X_prev = x          // input buffer
    X_next = new array   // output buffer

    l = n / 2
    m = 1
    for t = 1 to p:
        for j = 0 to l - 1:
            for k = 0 to m - 1:
                c0 = X_prev[k + j * m]
                c1 = X_prev[k + j * m + l * m]
                X_next[k + 2 * j * m]     = c0 + c1
                X_next[k + 2 * j * m + m] = ω(2l, j) * (c0 - c1)
        l = l / 2
        m = m * 2
        swap(X_prev, X_next)

    return X_prev
```

where `ω(2l, j) = e^(-2πij / (2l))`.

### Notes

- The Stockham algorithm is an **auto-sort** (or **self-sorting**) FFT: the output is in natural order without requiring a bit-reversal permutation.
- It uses **two buffers** that alternate roles each stage, trading memory for the elimination of the bit-reversal step.
- `l` tracks the half-size of the current stage, `m` tracks the stride between butterfly pairs.
- This formulation is well-suited for vectorization and GPU implementations.
