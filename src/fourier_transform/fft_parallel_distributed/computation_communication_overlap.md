# Computation-Communication Overlap Pattern

## Algorithm

A hybrid MPI+OpenMP pattern for overlapping communication with computation. The master thread performs MPI communication while the remaining threads execute independent computation in parallel. A second parallel loop then processes work that depends on the communication result.

### Pseudocode

```
function OverlapPattern(data, n):
    parallel region:

        // Master thread: MPI communication (non-blocking or blocking)
        if master_thread:
            MPI_Communication(...)

        // Worker threads: independent computation (dynamic schedule)
        parallel for i = 0 to n-1 (schedule: dynamic):
            Compute(data[i])        // no dependency on communication

        // barrier (implicit between omp for regions)

        // All threads: computation using communication result
        parallel for i = 0 to n-1:
            ComputeWithCommResult(data[i])
```

### Notes

- The `!$omp master` directive ensures only the master thread (thread 0) performs MPI communication — this avoids the need for `MPI_THREAD_MULTIPLE`.
- The first `!$omp do schedule(dynamic)` distributes independent work across all non-master threads. Dynamic scheduling handles load imbalance caused by the master thread being busy with communication.
- The implicit barrier at the end of each `!$omp do` ensures the communication is complete before the second loop begins.
- The second loop uses the default static schedule since all threads are available and the work is uniform.
- This pattern is used in parallel FFT implementations to overlap the all-to-all communication with local FFT computation on data that doesn't depend on the communicated values.
- Requires `MPI_Init_thread` with at least `MPI_THREAD_FUNNELED` (only master thread makes MPI calls).
