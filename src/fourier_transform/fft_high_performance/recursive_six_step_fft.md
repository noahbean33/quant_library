# Recursive Six-Step FFT Algorithm

## Algorithm

A cache-oblivious variant of the six-step FFT that recursively decomposes the problem until the data fits in cache, then falls back to a standard FFT. At each level, the length `n` is factored into `n1 × n2`, and the six-step procedure is applied with blocked transpositions.

### Pseudocode

```
function RecursiveFFT(a, b, w, n):
    // a: input/output array of n complex values
    // b: workspace array
    // w: twiddle factor array

    // Base case: fits in cache
    if n <= CACHE_SIZE:
        FFT(a, b, n)
        return

    // Step 0: Decompose n into n1 × n2
    (n1, n2) = factor(n)

    // Step 1: Blocked transposition (n1 × n2 → n2 × n1)
    for ii = 0 to n1-1 step nb:
        for jj = 0 to n2-1 step nb:
            for i = ii to min(ii+nb-1, n1-1):
                for j = jj to min(jj+nb-1, n2-1):
                    b[j + i*n2] = a[i + j*n1]

    // Step 2: n1 individual n2-point FFTs (recursive)
    for i = 0 to n1 - 1:
        RecursiveFFT(b[i*n2 .. (i+1)*n2-1], a, w, n2)

    // Step 3: Twiddle factor multiplication
    for i = 0 to n1*n2 - 1:
        a[i] = a[i] * w[i]

    // Step 4: Blocked transposition (n2 × n1 → n1 × n2)
    for jj = 0 to n2-1 step nb:
        for ii = 0 to n1-1 step nb:
            for j = jj to min(jj+nb-1, n2-1):
                for i = ii to min(ii+nb-1, n1-1):
                    b[i + j*n1] = a[j + i*n2]

    // Step 5: n2 individual n1-point FFTs (recursive)
    for j = 0 to n2 - 1:
        RecursiveFFT(b[j*n1 .. (j+1)*n1-1], a, w, n1)

    // Step 6: Blocked transposition (n1 × n2 → n2 × n1)
    for ii = 0 to n1-1 step nb:
        for jj = 0 to n2-1 step nb:
            for i = ii to min(ii+nb-1, n1-1):
                for j = jj to min(jj+nb-1, n2-1):
                    b[j + i*n2] = a[i + j*n1]
```

### Notes

- The recursion continues until the sub-problem fits in cache (`n <= CACHE_SIZE`), at which point a standard in-cache FFT is used.
- `factor(n)` decomposes `n` into two factors `n1` and `n2` — typically chosen to be as close to `√n` as possible for balanced recursion.
- This approach is **cache-oblivious**: it adapts to any cache hierarchy without needing to know the cache size at compile time (though `CACHE_SIZE` is used as a practical base case).
- The blocked transpositions use block size `nb` for cache efficiency at each recursion level.
