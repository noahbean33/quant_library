# Automatic Tuning of All-to-All Communication

## Algorithm

An empirical auto-tuning procedure that finds the optimal process grid decomposition `Px × Py = P` for all-to-all communication. When `P` processes need to perform an all-to-all exchange, it can be done as a single `MPI_Alltoall` or as a **two-step all-to-all** using a 2-D process grid. The optimal split depends on network topology, message size, and process count.

### Pseudocode

```
function AutotuneAlltoall(P, sendbuf, recvbuf, ITER_NUM):
    min_time = +∞

    for i = 0 to log2(P):
        Px = 2^i
        Py = P / Px

        MPI_Barrier()
        start = MPI_Wtime()

        repeat ITER_NUM times:
            if Px == 1 or Py == 1:
                MPI_Alltoall(sendbuf, recvbuf, ...)
            else:
                TwoStepAlltoall(sendbuf, recvbuf, Px, Py, ...)

        MPI_Barrier()
        end = MPI_Wtime()

        if (end - start) < min_time:
            min_time = end - start
            Qx = Px;  Qy = Py

    // Use optimal decomposition
    Px = Qx;  Py = Qy
```

### Notes

- When `Px = 1` or `Py = 1`, the decomposition is trivial and a standard `MPI_Alltoall` is used.
- The **two-step all-to-all** splits the global exchange into two phases along a 2-D process grid — this can reduce contention on networks with hierarchical topology (e.g., fat-tree, torus).
- The search space is small: only `log2(P) + 1` decompositions, each a power-of-two split.
- `MPI_Barrier` ensures synchronized timing across all processes.
- Multiple iterations amortize measurement noise.
- The optimal `(Px, Py)` is cached and reused for all subsequent all-to-all operations in the FFT.
- This tuning is especially important on large-scale systems where all-to-all is the dominant communication cost.
