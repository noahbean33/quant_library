# Cooley-Tukey FFT Algorithm (Iterative, In-Place)

## Algorithm

Given an input array `x` of `n = 2^m` complex values, compute the DFT in-place using the iterative Cooley-Tukey decimation-in-frequency approach, followed by bit-reversal permutation.

### Pseudocode

```
function FFT(x, n, m):
    // n = 2^m

    // Stage loop (decimation-in-frequency butterflies)
    l = n
    for k = 1 to m:
        px = -2π / l
        l = l / 2
        for j = 0 to l - 1:
            w = px * j
            for i = j to n - 1 step 2l:
                temp      = x[i] - x[i + l]
                x[i]      = x[i] + x[i + l]
                x[i + l]  = temp * (cos(w) + i·sin(w))

    // Bit-reversal permutation
    j = 0
    for i = 0 to n - 2:
        if i < j:
            swap(x[i], x[j])
        k = n / 2
        while k <= j:
            j = j - k
            k = k / 2
        j = j + k
```

### Notes

- The outer loop iterates over `m` stages, each halving the sub-problem size `l`.
- The twiddle factor is `e^(-2πij/l)` where `l` is the current stage size.
- After all butterfly stages, a **bit-reversal permutation** reorders the output into standard DFT order.
- This is an **in-place** algorithm requiring no additional storage beyond one temporary variable.
