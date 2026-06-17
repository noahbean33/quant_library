# Recursive Three-Step FFT Algorithm

## Algorithm

A cache-oblivious FFT that recursively decomposes a length-`n` transform into three steps: a butterfly with twiddle multiplication, two recursive half-length FFTs, and a transposition (interleave). The recursion bottoms out when the problem fits in cache.

### Pseudocode

```
function RecursiveFFT(a, temp, w, n):
    // a: input/output array of n complex values
    // temp: workspace array of n complex values
    // w: twiddle factors

    if n <= CACHE_SIZE:
        FFT(a, temp, w, n)       // base-case in-cache FFT
        return

    // Step 1: Butterfly + twiddle factor multiplication
    for i = 0 to n/2 - 1:
        temp[i]       = a[i] + a[i + n/2]
        temp[i + n/2] = (a[i] - a[i + n/2]) * w[i]

    // Step 2: Two recursive n/2-point FFTs
    RecursiveFFT(temp[0 .. n/2-1],       a, w[n/2 ..], n/2)
    RecursiveFFT(temp[n/2 .. n-1],       a, w[n/2 ..], n/2)

    // Step 3: Transposition (interleave even/odd halves)
    for i = 0 to n/2 - 1:
        a[2*i]     = temp[i]
        a[2*i + 1] = temp[i + n/2]
```

### Notes

- This is a **radix-2 Decimation-in-Frequency** decomposition expressed as three explicit steps.
- Step 1 combines the butterfly and twiddle multiplication into a single pass — this is the "multirow" kernel.
- Step 3 is equivalent to a perfect-shuffle (bit-reversal stride-2 interleave).
- The recursion naturally adapts to cache hierarchy: once `n <= CACHE_SIZE`, a standard FFT handles the sub-problem without further recursion overhead.
- This structure is the foundation for the OpenMP parallel variant, where each step becomes a parallel loop.
