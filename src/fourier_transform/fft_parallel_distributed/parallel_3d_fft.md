# Parallel Three-Dimensional FFT in Distributed-Memory Parallel Computers

## Algorithm

A distributed-memory parallel 3-D FFT for an `N1 × N2 × N3` complex array, block-distributed along the third dimension across `P` MPI processes. Each process holds `N1 × N2 × (N3/P)` elements. The algorithm performs FFTs along all three dimensions with two all-to-all communications: one after dimensions 1 and 2 are done (to redistribute for dimension 3), and one after dimension 3 (to restore the output distribution).

### Pseudocode

```
function Parallel3D_FFT(x_local, N1, N2, N3, P, rank):
    // x_local: N1 × N2 × N3_hat, where N3_hat = N3/P
    // Block distribution along dimension 3

    // Step 1: N2 * N3_hat individual N1-point FFTs
    for k = 0 to N3_hat - 1:
        for j = 0 to N2 - 1:
            FFT(x[0..N1-1, j, k], N1)

    // Step 2: Local transposition (swap dims 1 and 2)
    x2(J2, K1, J3_hat) = x1(K1, J2, J3_hat)

    // Step 3: N1 * N3_hat individual N2-point FFTs
    for k = 0 to N3_hat - 1:
        for k1 = 0 to N1 - 1:
            FFT(x[0..N2-1, k1, k], N2)

    // Step 4: Rearrangement
    Decompose K1 into (K1_tilde, P1)
    Rearrange x3 into (K1_tilde, K2, J3_hat, P1)

    // Step 5: All-to-all communication
    MPI_Alltoall(...)
    // Result: (K1_hat, K2, J3_tilde, P3)

    // Step 6: Local transposition
    Assemble full J3 from (J3_tilde, P3)
    Transpose to (J3, K1_hat, K2)

    // Step 7: N1_hat * N2 individual N3-point FFTs
    for k2 = 0 to N2 - 1:
        for k1h = 0 to N1_hat - 1:
            FFT(x[0..N3-1, k1h, k2], N3)

    // Step 8: Local transposition
    Transpose: x8(K1_hat, K2, K3) = x7(K3, K1_hat, K2)
    Decompose K3 into (K3_tilde, P3)

    // Step 9: All-to-all communication
    MPI_Alltoall(...)
    // Result: (K1_tilde, K2, K3_hat, P1)

    // Step 10: Rearrangement → final output
    Assemble full K1 from (K1_tilde, P1)
    y_local(K1, K2, K3_hat)
```

### Notes

- **Two all-to-all communications** (steps 5 and 9) redistribute data between the local and distributed dimensions.
- Dimensions 1 and 2 are fully local before the first all-to-all, so their FFTs (steps 1, 3) require no communication.
- After step 5, dimension 3 becomes fully local (assembled from all processes), enabling the N3-point FFTs in step 7.
- The second all-to-all (step 9) restores block distribution along the third dimension for the output.
- `N3` must be divisible by `P`; `N1` must also be divisible by `P` for the redistribution.
- Local transpositions (steps 2, 6, 8) ensure contiguous memory access for the FFTs along each dimension.
