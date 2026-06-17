# Radix-5 FFT Algorithm (Stockham Auto-Sort)

## Algorithm

Given an input array `x` of `n = 5^p` complex values, compute the DFT using a radix-5 Stockham-style decomposition. Uses two alternating buffers and produces output in natural order.

### Constants

- `ω_q = e^(-2πi/q)` — primitive q-th root of unity
- `sin(2π/5) ≈ 0.95105651629`
- `sin(π/5) ≈ 0.58778525229`
- `√5/4 ≈ 0.55901699437`

### Pseudocode

```
function Radix5_FFT(x, n):
    // n = 5^p
    X_prev = x
    X_next = new array[n]

    l = n / 5
    m = 1

    for t = 1 to p:
        for j = 0 to l - 1:
            ω = e^(-2πij / (5l))
            for k = 0 to m - 1:
                c0 = X_prev[k + j*m]
                c1 = X_prev[k + j*m + l*m]
                c2 = X_prev[k + j*m + 2*l*m]
                c3 = X_prev[k + j*m + 3*l*m]
                c4 = X_prev[k + j*m + 4*l*m]

                // Symmetric / antisymmetric combinations
                d0 = c1 + c4
                d1 = c2 + c3
                d2 = sin(2π/5) * (c1 - c4)
                d3 = sin(2π/5) * (c2 - c3)
                d4 = d0 + d1
                d5 = (√5/4) * (d0 - d1)
                d6 = c0 - (1/4) * d4
                d7 = d6 + d5
                d8 = d6 - d5
                d9  = -i * (d2 + sin(π/5)/sin(2π/5) * d3)
                d10 = -i * (sin(π/5)/sin(2π/5) * d2 - d3)

                X_next[k + 5*j*m]       = c0 + d4
                X_next[k + 5*j*m + m]   = ω^j   * (d7 + d9)
                X_next[k + 5*j*m + 2*m] = ω^(2j) * (d8 + d10)
                X_next[k + 5*j*m + 3*m] = ω^(3j) * (d8 - d10)
                X_next[k + 5*j*m + 4*m] = ω^(4j) * (d7 - d9)

        l = l / 5
        m = m * 5
        swap(X_prev, X_next)

    return X_prev
```

### Notes

- The length-5 DFT butterfly is decomposed into real arithmetic using the specific sine values of the 5th roots of unity.
- The ratio `sin(π/5) / sin(2π/5)` appears naturally from the Rader/Winograd decomposition of the length-5 DFT.
- The outputs are arranged so that `d7 ± d9` and `d8 ± d10` pair up conjugate-symmetric frequency bins.
- This is a **self-sorting** (Stockham) formulation — no digit-reversal permutation is needed.
