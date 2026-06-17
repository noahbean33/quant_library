# Radix-3 FFT Algorithm (Stockham Auto-Sort)

## Algorithm

Given an input array `x` of `n = 3^p` complex values, compute the DFT using a radix-3 Stockham-style decomposition. Uses two buffers that alternate each stage, producing output in natural order.

### Constants

- `ω_q = e^(-2πi/q)` — primitive q-th root of unity
- `sin(π/3) = √3/2 ≈ 0.86602540378`

### Pseudocode

```
function Radix3_FFT(x, n):
    // n = 3^p
    X_prev = x
    X_next = new array[n]

    l = n / 3
    m = 1

    for t = 1 to p:
        for j = 0 to l - 1:
            ω = e^(-2πij / (3l))
            for k = 0 to m - 1:
                c0 = X_prev[k + j*m]
                c1 = X_prev[k + j*m + l*m]
                c2 = X_prev[k + j*m + 2*l*m]

                d0 = c1 + c2
                d1 = c0 - (1/2) * d0
                d2 = -i * sin(π/3) * (c1 - c2)

                X_next[k + 3*j*m]       = c0 + d0
                X_next[k + 3*j*m + m]   = ω^j   * (d1 + d2)
                X_next[k + 3*j*m + 2*m] = ω^(2j) * (d1 - d2)

        l = l / 3
        m = m * 3
        swap(X_prev, X_next)

    return X_prev
```

### Notes

- The intermediate values `d1 ± d2` correspond to the two non-trivial outputs of a length-3 DFT butterfly.
- `d2` contains the imaginary factor `-i·sin(π/3)·(c1 - c2)`, which encodes the 120° phase shifts of the 3rd roots of unity.
- This is a **self-sorting** (Stockham) formulation — no digit-reversal permutation is needed.
