# Radix-8 FFT Algorithm (Stockham Auto-Sort)

## Algorithm

Given an input array `x` of `n = 8^p` complex values, compute the DFT using a radix-8 Stockham-style decomposition. Uses two alternating buffers and produces output in natural order.

### Constants

- `ω_q = e^(-2πi/q)` — primitive q-th root of unity
- `√2/2 ≈ 0.70710678118`

### Pseudocode

```
function Radix8_FFT(x, n):
    // n = 8^p
    X_prev = x
    X_next = new array[n]

    l = n / 8
    m = 1

    for t = 1 to p:
        for j = 0 to l - 1:
            ω = e^(-2πij / (8l))
            for k = 0 to m - 1:
                c0 = X_prev[k + j*m]
                c1 = X_prev[k + j*m + l*m]
                c2 = X_prev[k + j*m + 2*l*m]
                c3 = X_prev[k + j*m + 3*l*m]
                c4 = X_prev[k + j*m + 4*l*m]
                c5 = X_prev[k + j*m + 5*l*m]
                c6 = X_prev[k + j*m + 6*l*m]
                c7 = X_prev[k + j*m + 7*l*m]

                // Stage 1: radix-2 pairs
                d0 = c0 + c4
                d1 = c0 - c4
                d2 = c2 + c6
                d3 = -i * (c2 - c6)
                d4 = c1 + c5
                d5 = c1 - c5
                d6 = c3 + c7
                d7 = c3 - c7

                // Stage 2: radix-4 combinations
                e0 = d0 + d2
                e1 = d0 - d2
                e2 = d4 + d6
                e3 = -i * (d4 - d6)
                e4 = (√2/2) * (d5 - d7)
                e5 = -(√2/2) * i * (d5 + d7)
                e6 = d1 + e4
                e7 = d1 - e4
                e8 = d3 + e5
                e9 = d3 - e5

                // Stage 3: output with twiddle factors
                X_next[k + 8*j*m]       = e0 + e2
                X_next[k + 8*j*m + m]   = ω^j   * (e6 + e8)
                X_next[k + 8*j*m + 2*m] = ω^(2j) * (e1 + e3)
                X_next[k + 8*j*m + 3*m] = ω^(3j) * (e7 - e9)
                X_next[k + 8*j*m + 4*m] = ω^(4j) * (e0 - e2)
                X_next[k + 8*j*m + 5*m] = ω^(5j) * (e7 + e9)
                X_next[k + 8*j*m + 6*m] = ω^(6j) * (e1 - e3)
                X_next[k + 8*j*m + 7*m] = ω^(7j) * (e6 - e8)

        l = l / 8
        m = m * 8
        swap(X_prev, X_next)

    return X_prev
```

### Notes

- The radix-8 butterfly is decomposed into three stages: two levels of radix-2 operations followed by the final output combination.
- The factor `√2/2` in `e4` and `e5` comes from the 8th roots of unity at positions ±π/4.
- Radix-8 achieves further multiplication savings over radix-4 by exploiting trivial twiddle factors at multiples of π/4.
- The output ordering (0, 1, 2, 3, 4, 5, 6, 7) maps to DFT indices via the standard digit-reversal pattern, but the Stockham formulation handles this implicitly — **no digit-reversal permutation is needed**.
