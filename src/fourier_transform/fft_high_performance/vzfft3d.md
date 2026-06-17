# 3-D Complex FFT Routine (for Vector Machines)

## Algorithm

Computes a 3-D complex FFT of size `nx × ny × nz` using a **row-column approach** optimized for vector machines. Each dimension is transformed sequentially using mixed-radix multi-column FFTs, with transpositions between each dimension to maintain contiguous memory access.

### Interface

```
vzfft3d(a, nx, ny, nz, iopt)
```

- **a[nx × ny × nz]**: complex input/output array
- **nx, ny, nz**: transform lengths (`each = 2^p * 3^q * 5^r`)
- **iopt**: `0` = initialize, `-1` = forward FFT, `+1` = inverse FFT

### Pseudocode

```
function VZFFT3D(a, nx, ny, nz, iopt):
    if iopt == 0:
        initialize twiddle tables wx, wy, wz
        return

    if iopt == +1:
        conjugate all elements of a

    // Step 1: (nx·ny) multi-column nz-point FFTs (z-dimension)
    MFFT235A(a, b, wz, nx·ny, nz)

    // Step 2: Transpose (nx·ny × nz → nz × nx·ny)
    ZTRANS(a, b, nx·ny, nz)

    // Step 3: (nz·nx) multi-column ny-point FFTs (y-dimension)
    MFFT235A(b, a, wy, nz·nx, ny)

    // Step 4: Transpose (nz·nx × ny → ny × nz·nx)
    ZTRANS(b, a, nz·nx, ny)

    // Step 5: (ny·nz) multi-column nx-point FFTs (x-dimension)
    MFFT235B(a, b, wx, ny·nz, nx)

    // Step 6: Transpose (ny·nz × nx → nx × ny·nz)
    ZTRANS(b, a, ny·nz, nx)

    if iopt == +1:
        conjugate and scale a by 1/(nx·ny·nz)
```

### Notes

- The 3-D FFT is decomposed into three passes of 1-D multi-column FFTs along each dimension (z → y → x), separated by transpositions.
- The transpositions convert strided access into contiguous access, which is critical for **vector machine** performance.
- This variant does not use OpenMP; it relies on long vectorizable loops.
- The inverse transform uses the conjugation identity: `IFFT₃D(x) = conj(FFT₃D(conj(x))) / (nx·ny·nz)`.
