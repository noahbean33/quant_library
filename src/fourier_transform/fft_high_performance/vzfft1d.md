# 1-D Complex FFT Routine (for Vector Machines)

## Algorithm

Computes a 1-D complex FFT of length `n = nx * ny` where `n = 2^p * 3^q * 5^r`, using the **six-step FFT** approach optimized for vector machines. The data is viewed as an `nx × ny` matrix and transformed using multi-column FFTs with twiddle factor multiplication and transposition.

### Interface

```
vzfft1d(a, n, iopt, b, wx, wy)
```

- **a[n]**: complex input/output vector
- **b[2n]**: work/coefficient vector
- **n**: transform length (`n = 2^p * 3^q * 5^r`)
- **iopt**: `0` = initialize coefficients, `-1` = forward FFT, `+1` = inverse FFT

### Pseudocode

```
function VZFFT1D(a, n, iopt, b):
    if iopt == +1:
        conjugate all elements of a

    factor n into nx * ny  (nx ≈ √n)

    if iopt == 0:
        initialize twiddle tables wx, wy
        compute twiddle matrix w[i][j] = e^(-2πi·i·j / n) and store in b[n:]
        return

    // Six-step FFT for vector machines:
    // Step 1-2: nx multi-column ny-point FFTs
    MFFT235A(a, b, wy, nx, ny)

    // Step 3-4: Transpose with twiddle factor multiplication (nx × ny → ny × nx)
    ZTRANSMUL(a, b, w, nx, ny)

    // Step 5-6: ny multi-column nx-point FFTs
    MFFT235B(b, a, wx, ny, nx)

    if iopt == +1:
        conjugate and scale a by 1/n
```

### Transpose with Twiddle Multiplication

The `ZTRANSMUL` routine combines matrix transposition with element-wise twiddle factor multiplication. Two strategies are used:

- **ZTRANSMULA**: Simple row-by-row transpose-multiply, suited for non-power-of-2 dimensions.
- **ZTRANSMULB**: Diagonal traversal transpose-multiply, which avoids cache conflicts when dimensions are powers of 2.

### Notes

- This variant is optimized for **vector machines** (no OpenMP directives; relies on vectorizable loops).
- The six-step decomposition `n = nx × ny` converts the 1-D FFT into two sets of shorter multi-column FFTs separated by a transpose+twiddle step.
- The factorization `GETNXNY` selects `nx ≈ √n` so that both `nx` and `ny` are smooth numbers (products of 2, 3, 5).
- The inverse transform is computed via conjugation: `IFFT(x) = conj(FFT(conj(x))) / n`.
