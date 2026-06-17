# OpenMP Implementation of a Recursive Three-Step FFT Algorithm

## Algorithm

A shared-memory parallel FFT that adds OpenMP directives to the recursive three-step algorithm. Each of the three steps — butterfly+twiddle, recursive sub-FFTs, and interleave — is parallelized with `#pragma omp parallel for`.

### Pseudocode

```
function ParallelFFT(a, temp, w, n):
    // a: input/output array of n complex values
    // temp: workspace array of n complex values
    // w: twiddle factors

    if n <= CACHE_SIZE:
        FFT(a, temp, w, n)
        return

    // Step 1: Butterfly + twiddle (parallel)
    parallel for i = 0 to n/2 - 1:
        temp[i]       = a[i] + a[i + n/2]
        temp[i + n/2] = (a[i] - a[i + n/2]) * w[i]

    // Step 2: Two recursive n/2-point FFTs (parallel tasks)
    parallel for j = 0 to 1:
        RecursiveFFT(temp[j*(n/2) .. (j+1)*(n/2)-1],
                     a[j*(n/2) .. (j+1)*(n/2)-1],
                     w[n/2 ..], n/2)

    // Step 3: Interleave (parallel)
    parallel for i = 0 to n/2 - 1:
        a[2*i]     = temp[i]
        a[2*i + 1] = temp[i + n/2]
```

### Notes

- The three `parallel for` regions are enclosed in a single `parallel` block with implicit barriers between steps, ensuring correctness.
- Step 2 has only 2 iterations, so parallelism there is coarse-grained; the real benefit comes from steps 1 and 3 which have `n/2` independent iterations.
- At deeper recursion levels, the sub-FFTs call the serial `recursive_fft` — only the top level(s) use OpenMP to avoid thread-creation overhead.
- This pattern generalizes: for a radix-`r` decomposition, step 2 would have `r` independent tasks.
