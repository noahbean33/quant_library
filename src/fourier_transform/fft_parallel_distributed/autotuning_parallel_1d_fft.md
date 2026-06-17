# Automatic Tuning of Parallel One-Dimensional FFT

## Algorithm

An empirical auto-tuning procedure that searches over the parameter space of a parallel 1-D FFT to find the configuration that minimizes wall-clock time. The tunable parameters are:

- **`NB`** — cache block size (`2^k` for `k = 2..6`, i.e., 4 to 64)
- **`N1, N2`** — factorization of `N = N1 × N2` (with `N1 = 2^j` for `j = log2(P)..log2(√N)`)
- **`NDIV`** — communication subdivision factor (`GCD(i, GCD(N1/P, N2/P))` for `i = 1..16`)

The algorithm performs a grid search, timing each configuration over multiple iterations, and selects the fastest.

### Pseudocode

```
function AutotuneParallel1DFFT(N, P):
    min_time = +∞

    for k = 2 to 6:
        NB = 2^k

        for j = log2(P) to log2(√N):
            N1 = 2^j
            N2 = N / N1

            for i = 1 to 16:
                NDIV = GCD(i, GCD(N1/P, N2/P))

                MPI_Barrier()
                start = MPI_Wtime()

                repeat ITER_NUM times:
                    Parallel_1D_FFT(N1, N2, NB, NDIV, ...)

                MPI_Barrier()
                end = MPI_Wtime()

                if (end - start) < min_time:
                    min_time = end - start
                    M1 = N1;  M2 = N2;  MB = NB;  MDIV = NDIV

    // Use optimal parameters
    N1 = M1;  N2 = M2;  NB = MB;  NDIV = MDIV
```

### Notes

- The search space is kept tractable: `NB` has 5 values, `N1` has `O(log N)` values, and `NDIV` has ≤ 16 values.
- `MPI_Barrier` before and after the timing loop ensures all processes measure the same interval.
- Multiple iterations (`ITER_NUM`) are timed to amortize measurement noise.
- `NDIV` controls how the all-to-all communication is subdivided — larger values can improve overlap with computation but increase message count.
- The `GCD` constraint ensures `NDIV` divides both `N1/P` and `N2/P`, guaranteeing even data distribution.
- This auto-tuning is performed once during initialization; the optimal parameters are then reused for all subsequent FFT calls.
