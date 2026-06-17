# Parallel One-Dimensional FFT Using Block Distribution

## Algorithm

A distributed-memory parallel 1-D FFT based on the six-step decomposition. The data of length `N = N1 × N2` is viewed as an `N1 × N2` matrix and block-distributed along the second dimension across `P` MPI processes. Each process holds `N1 × (N2/P)` elements. The algorithm uses three all-to-all communications interspersed with local FFTs, transpositions, and twiddle factor multiplications.

### Pseudocode

```
function Parallel1D_FFT_Block(x_local, N1, N2, P, rank):
    // x_local: local portion N1 × N2_hat, where N2_hat = N2/P
    // Block distribution: J = rank * N_hat + J_hat

    // Step 1: Rearrangement (local reshape for all-to-all)
    Rearrange x_local from (N1, N2_hat) into (N1_tilde, N2_hat, P1)

    // Step 2: All-to-all communication
    MPI_Alltoall(...)
    // Result: (N1_hat, N2_tilde, P2)

    // Step 3: Local transposition
    Transpose to get (J2, J1_hat) layout

    // Step 4: N1/P individual N2-point multicolumn FFTs
    for each local column j1_hat = 0 to N1_hat - 1:
        FFT(x_local[0..N2-1, j1_hat], N2)

    // Step 5: Twiddle factor multiplication + rearrangement
    for j1_hat, k2:
        x_local[k2, j1_hat] *= ω_N^(j1_hat * k2)
    Rearrange into (K2_tilde, J1_hat, P2)

    // Step 6: All-to-all communication
    MPI_Alltoall(...)

    // Step 7: Local transposition
    Transpose to get (J1, K2_hat) layout

    // Step 8: N2/P individual N1-point multicolumn FFTs
    for each local column k2_hat = 0 to N2_hat - 1:
        FFT(x_local[0..N1-1, k2_hat], N1)

    // Step 9: Rearrangement
    Rearrange into (K1_tilde, K2_hat, P1)

    // Step 10: All-to-all communication
    MPI_Alltoall(...)

    // Step 11: Local transposition → final output
    Transpose to get y_local(K2, K1_hat)
```

### Notes

- The data is **block-distributed**: process `m` owns global indices `[m*N_hat, (m+1)*N_hat - 1]`.
- **Three all-to-all communications** are required (steps 2, 6, 10), which is the main communication cost.
- Steps 4 and 8 perform `√N/P` individual `√N`-point FFTs when `N1 = N2 = √N`.
- Step 5 combines twiddle factor multiplication with data rearrangement to reduce memory bandwidth usage.
- Both `N1` and `N2` must be divisible by `P` for uniform load balancing.
