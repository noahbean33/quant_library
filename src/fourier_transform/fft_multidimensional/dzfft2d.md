# 2-D Real-to-Complex FFT Routine (with OpenMP and Cache Blocking)

## Algorithm

Computes a 2-D real-to-complex FFT of an `nx × ny` real array. The output exploits **conjugate symmetry** of the real-input DFT, storing only `(nx/2 + 1) × ny` complex values. Uses a packed-pair technique to halve the number of x-direction FFTs.

### Interface

```
dzfft2d(a, nx, ny, iopt, b)
```

- **a[nx × ny]**: real input / complex output (half-complex format: `(nx/2+1) × ny`)
- **b[(nx/2+1) × ny]**: work vector
- **nx, ny**: transform lengths (`each = 2^p * 3^q * 5^r`)
- **iopt**: `0` = initialize, `-1` = forward transform

### Pseudocode

```
function DZFFT2D(da, a, nx, ny, iopt):
    if iopt == 0:
        SETTBL(wx, nx), SETTBL(wy, ny)
        return

    // Phase 1: Packed real-to-complex x-direction FFTs
    for j = 0 to ny-1 step 2:
        // Pack two real rows into one complex vector
        cx[i] = da[i][j] + i·da[i][j+1]   for i = 0..nx-1
        FFT235(cx, wx, nx)

        // Unpack using conjugate symmetry:
        b[0][j]   = Re(cx[0])
        b[0][j+1] = Im(cx[0])
        for i = 1 to nx/2:
            b[i][j]   = ½ (cx[i] + conj(cx[nx-i]))
            b[i][j+1] = -½i (cx[i] - conj(cx[nx-i]))

    // (Handle odd ny: last row done separately as real-only FFT)

    // Phase 2: Blocked ny-point FFTs on each frequency bin
    for i = 0 to nx/2:
        extract column b[i][0..ny-1]
        FFT235(column, wy, ny)
        store into a[i][0..ny-1]
```

### Notes

- The **packed-pair technique** processes two real rows simultaneously by packing them as real and imaginary parts of a complex vector, then separating the results using conjugate symmetry.
- This halves the number of x-direction FFTs compared to a naive real-to-complex approach.
- The output has **half-complex** format: only `nx/2 + 1` frequency bins along x due to conjugate symmetry (`X[nx-k] = conj(X[k])` for real input).
- Cache blocking (`NBLK = 16`) is applied to the y-direction FFT phase.
- This is the FFTE library's `DZFFT2D` routine.
