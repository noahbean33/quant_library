# 2-D Complex FFT Routine (for Vector Machines)

## Algorithm

Computes a 2-D complex FFT of size `nx × ny` using a **row-column approach** optimized for vector machines. Each dimension is transformed using mixed-radix multi-column FFTs, separated by transpositions to maintain contiguous memory access.

### Interface

```
vzfft2d(a, nx, ny, iopt)
```

- **a[nx × ny]**: complex input/output matrix (row-major)
- **nx, ny**: transform lengths (`nx = 2^p * 3^q * 5^r`, similarly for `ny`)
- **iopt**: `0` = initialize, `-1` = forward FFT, `+1` = inverse FFT

### Pseudocode

```
function VZFFT2D(a, nx, ny, iopt):
    if iopt == 0:
        initialize twiddle tables wx, wy
        return

    if iopt == +1:
        conjugate all elements of a

    // Step 1: nx multi-column ny-point FFTs (along y-dimension)
    MFFT235A(a, b, wy, nx, ny)

    // Step 2: Transpose (nx × ny → ny × nx)
    ZTRANS(a, b, nx, ny)

    // Step 3: ny multi-column nx-point FFTs (along x-dimension)
    MFFT235A(b, a, wx, ny, nx)

    // Step 4: Transpose (ny × nx → nx × ny)
    ZTRANS(b, a, ny, nx)

    if iopt == +1:
        conjugate and scale a by 1/(nx·ny)
```

### Notes

- Uses allocatable workspace `b` of size `nx × ny` for transpositions.
- The **row-column decomposition** transforms each dimension independently, with transpositions ensuring column-wise (contiguous) access patterns for vectorization.
- This is the vector-machine variant — it relies on long vector lengths rather than OpenMP thread parallelism.
- The inverse transform uses the conjugation identity: `IFFT₂D(x) = conj(FFT₂D(conj(x))) / (nx·ny)`.
