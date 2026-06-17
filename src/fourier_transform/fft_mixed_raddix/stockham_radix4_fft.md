# Radix-4 FFT Algorithm (Stockham Auto-Sort)

## Algorithm

Given an input array `x` of `n = 4^p` complex values, compute the DFT using a radix-4 Stockham-style decomposition. Uses two alternating buffers and produces output in natural order.

### Constants

- `ω_q = e^(-2πi/q)` — primitive q-th root of unity

### Pseudocode

```
function Radix4_FFT(x, n):
    // n = 4^p
    X_prev = x
    X_next = new array[n]

    l = n / 4
    m = 1

    for t = 1 to p:
        for j = 0 to l - 1:
            ω = e^(-2πij / (4l))
            for k = 0 to m - 1:
                c0 = X_prev[k + j*m]
                c1 = X_prev[k + j*m + l*m]
                c2 = X_prev[k + j*m + 2*l*m]
                c3 = X_prev[k + j*m + 3*l*m]

                d0 = c0 + c2
                d1 = c0 - c2
                d2 = c1 + c3
                d3 = -i * (c1 - c3)

                X_next[k + 4*j*m]       = d0 + d2
                X_next[k + 4*j*m + m]   = ω^j   * (d1 + d3)
                X_next[k + 4*j*m + 2*m] = ω^(2j) * (d0 - d2)
                X_next[k + 4*j*m + 3*m] = ω^(3j) * (d1 - d3)

        l = l / 4
        m = m * 4
        swap(X_prev, X_next)

    return X_prev
```

### Notes

- The radix-4 butterfly computes a length-4 DFT: the four outputs are linear combinations of `d0 ± d2` and `d1 ± d3`, where `d3` includes the factor `-i` representing the 4th root of unity.
- Radix-4 requires **25% fewer multiplications** than radix-2 for the same problem size, since every other twiddle factor stage is trivial.
- This is a **self-sorting** (Stockham) formulation — no bit-reversal permutation is needed.
