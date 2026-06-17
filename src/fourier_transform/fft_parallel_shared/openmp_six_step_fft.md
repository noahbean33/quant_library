# Parallelization of Six-Step FFT with Loop Collapse

## Algorithm

An OpenMP-parallelized version of the blocked six-step FFT. The `collapse(2)` clause is used on the nested blocked transposition loops to increase the number of parallel iterations and improve load balancing. The column FFTs in steps 2 and 5 are also parallelized with `#pragma omp for`.

### Pseudocode

```
function OpenMP_SixStepFFT(x, y, w, n1, n2, nb):
    // x: n1 × n2 complex matrix
    // y: n2 × n1 complex matrix (output)
    // w: n1 × n2 twiddle factors
    // nb: cache block size

    parallel:

        // Step 1: Blocked transposition (collapse(2) on tile loops)
        parallel for collapse(2) ii = 0..n1-1 step nb, jj = 0..n2-1 step nb:
            for i = ii to min(ii+nb-1, n1-1):
                for j = jj to min(jj+nb-1, n2-1):
                    y[j][i] = x[i][j]

        // Step 2: n1 individual n2-point FFTs
        parallel for i = 0 to n1 - 1:
            FFT(y[0..n2-1][i], n2)

        // Steps 3-4: Twiddle multiplication + blocked transposition (collapse(2))
        parallel for collapse(2) jj = 0..n2-1 step nb, ii = 0..n1-1 step nb:
            for j = jj to min(jj+nb-1, n2-1):
                for i = ii to min(ii+nb-1, n1-1):
                    x[i][j] = y[j][i] * w[i][j]

        // Step 5: n2 individual n1-point FFTs
        parallel for j = 0 to n2 - 1:
            FFT(x[0..n1-1][j], n1)

        // Step 6: Blocked transposition (collapse(2))
        parallel for collapse(2) ii = 0..n1-1 step nb, jj = 0..n2-1 step nb:
            for i = ii to min(ii+nb-1, n1-1):
                for j = jj to min(jj+nb-1, n2-1):
                    y[j][i] = x[i][j]
```

### Notes

- **`collapse(2)`** merges the two tile-index loops (`ii`, `jj`) into a single parallel iteration space, providing `(n1/nb) × (n2/nb)` independent chunks instead of just `n1/nb`. This dramatically improves load balancing for large thread counts.
- All six steps are enclosed in a single `#pragma omp parallel` region with implicit barriers between each `#pragma omp for`, avoiding repeated fork/join overhead.
- The FFT loops (steps 2 and 5) have `n1` and `n2` independent iterations respectively — naturally parallel.
- This is Takahashi's recommended approach for shared-memory parallelization of the six-step FFT on multicore CPUs.
