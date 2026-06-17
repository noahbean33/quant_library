# Parallel Two-Dimensional FFT in Distributed-Memory Parallel Computers

## Algorithm

A distributed-memory parallel 2-D FFT for an `N1 × N2` complex array, block-distributed along the second dimension across `P` MPI processes. Each process holds `N1 × (N2/P)` elements. The algorithm performs FFTs along each dimension with all-to-all communications to redistribute data between dimensions.

### Pseudocode

```
function Parallel2D_FFT(x_local, N1, N2, P, rank):
    // x_local: N1 × N2_hat, where N2_hat = N2/P
    // Block distribution along dimension 2

    // Step 1: (N2/P) individual N1-point multicolumn FFTs
    for j2_hat = 0 to N2_hat - 1:
        FFT(x_local[0..N1-1, j2_hat], N1)

    // Step 2: Local transposition
    Transpose: x2(J2_hat, K1) = x1(K1, J2_hat)
    Decompose K1 into (K1_tilde, P1) for all-to-all

    // Step 3: All-to-all communication
    MPI_Alltoall(...)
    // Result: (J2_tilde, K1_hat, P2)

    // Step 4: Rearrangement
    Assemble full J2 from (J2_tilde, P2) → layout (J2, K1_hat)

    // Step 5: (N1/P) individual N2-point multicolumn FFTs
    for k1_hat = 0 to N1_hat - 1:
        FFT(x_local[0..N2-1, k1_hat], N2)

    // Step 6: Local transposition
    Transpose: x6(K1_hat, K2) = x5(K2, K1_hat)
    Decompose K2 into (K2_tilde, P2)

    // Step 7: All-to-all communication
    MPI_Alltoall(...)
    // Result: (K1_tilde, K2_hat, P1)

    // Step 8: Rearrangement → final output
    Assemble full K1 from (K1_tilde, P1) → y_local(K1, K2_hat)
```

### Notes

- **Two all-to-all communications** are required — one after each dimension's FFT — to redistribute data so the next dimension becomes local.
- Steps 1 and 5 perform `N2/P` and `N1/P` independent FFTs respectively, which are embarrassingly parallel within each process.
- The transpositions (steps 2, 6) and rearrangements (steps 4, 8) are local operations that reorder data for the communication or the next FFT phase.
- `N2` must be divisible by `P` for the initial block distribution; `N1` must also be divisible by `P` for the redistribution to work evenly.
