# Parallel One-Dimensional FFT Using Cyclic Distribution

## Algorithm

A distributed-memory parallel 1-D FFT based on the six-step decomposition with **cyclic distribution**. The data of length `N = N1 × N2` is viewed as an `N1 × N2` matrix and cyclic-distributed along the first dimension across `P` MPI processes. Each process holds `(N1/P) × N2` elements. This variant requires only **one all-to-all communication** (versus three for block distribution), and both input and output remain in natural order.

### Pseudocode

```
function Parallel1D_FFT_Cyclic(x_local, N1, N2, P, rank):
    // x_local: local portion N1_hat × N2, where N1_hat = N1/P
    // Cyclic distribution: J = J_hat * P + rank

    // Step 1: Local transposition
    Transpose x_local from (N1_hat, N2) to (J2, J1_hat)

    // Step 2: N1_hat individual N2-point multicolumn FFTs
    for j1_hat = 0 to N1_hat - 1:
        FFT(x_local[0..N2-1, j1_hat], N2)

    // Step 3: Twiddle factor multiplication + transposition
    for j1_hat, k2:
        x[j1_hat, k2] = FFT_result[k2, j1_hat] * ω_N^(j1_hat * k2)
    Rearrange into (J1_hat, P2, K2_tilde)

    // Step 4: Rearrangement
    Rearrange into (J1_hat, K2_tilde, P2) for all-to-all

    // Step 5: All-to-all communication
    MPI_Alltoall(...)
    // Result: (J1_tilde, K2_hat, P1)

    // Step 6: Transposition
    Assemble global J1 from (P1, J1_tilde) → layout (J1, K2_hat)

    // Step 7: N2_hat individual N1-point multicolumn FFTs
    for k2_hat = 0 to N2_hat - 1:
        FFT(x_local[0..N1-1, k2_hat], N1)

    // Step 8: Transposition → final output
    Transpose to y_local(K2_hat, K1)
```

### Notes

- **Cyclic distribution**: process `m` owns global index `J = J_hat * P + m`, giving interleaved ownership.
- Only **one all-to-all communication** (step 5) is needed — a major advantage over the block distribution variant which requires three.
- Both input `x` and output `y` are in **natural order**, simplifying integration with other application code.
- Step 3 combines twiddle multiplication with transposition for memory bandwidth efficiency.
- When `N1 = N2 = √N`, each process performs `√N/P` individual `√N`-point FFTs in steps 2 and 7.
- Requires both `N1` and `N2` to be divisible by `P`.
