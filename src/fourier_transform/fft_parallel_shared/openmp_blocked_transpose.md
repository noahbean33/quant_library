# OpenMP Cache-Blocked Matrix Transposition

## Algorithm

Transposes an `n1 × n2` complex matrix into an `n2 × n1` matrix using cache blocking and OpenMP parallelism. The outer loop tiles are distributed across threads, and the inner blocked copy ensures data stays in cache during the transpose.

### Pseudocode

```
function BlockedTranspose(x, y, n1, n2, nb):
    // x: n1 × n2 input matrix
    // y: n2 × n1 output matrix
    // nb: cache block size

    parallel for ii = 0 to n1-1 step nb:
        for jj = 0 to n2-1 step nb:
            for i = ii to min(ii + nb - 1, n1 - 1):
                for j = jj to min(jj + nb - 1, n2 - 1):
                    y[j][i] = x[i][j]
```

### Notes

- The block size `nb` is chosen so that an `nb × nb` tile fits in L1/L2 cache, minimizing TLB misses and cache-line evictions.
- The `parallel for` on the outer `ii` loop distributes tile rows across OpenMP threads.
- This is a fundamental building block for high-performance FFTs — the six-step FFT, for instance, uses this pattern in steps 1, 4, and 6.
- For power-of-two dimensions, padding may be added to avoid cache-line conflicts (same `np` padding seen in the FFT workspace arrays).
