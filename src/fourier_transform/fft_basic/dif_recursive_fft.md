# Decimation-in-Frequency FFT (Recursive)

## Algorithm

Given an input array `x` of `n` complex values (where `n` is a power of 2), compute the DFT in-place using decimation-in-frequency decomposition.

### Pseudocode

```
function FFT(x, temp, n):
    if n <= 1:
        return

    ω_n = e^(-2πi / n)

    // Butterfly: combine first and second halves
    for j = 0 to n/2 - 1:
        w = -2π * j / n
        temp[j]       = x[j] + x[j + n/2]
        temp[j + n/2] = (x[j] - x[j + n/2]) * (cos(w) + i·sin(w))

    // Recurse on each half
    FFT(temp[0 .. n/2-1],       x, n/2)
    FFT(temp[n/2 .. n-1],       x, n/2)

    // Interleave (unshuffle) results back into x
    for j = 0 to n/2 - 1:
        x[2j]     = temp[j]
        x[2j + 1] = temp[j + n/2]
```

### Notes

- The twiddle factor is `e^(-2πij/n) = cos(-2πj/n) + i·sin(-2πj/n)`.
- After the two recursive half-size FFTs, the even-indexed and odd-indexed outputs are interleaved to produce the final result.
- This is the **DIF (decimation-in-frequency)** variant of the Cooley-Tukey algorithm.
