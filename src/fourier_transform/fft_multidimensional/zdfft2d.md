# 2-D Complex-to-Real FFT Routine (with OpenMP and Cache Blocking)

## Algorithm

Computes a 2-D complex-to-real (inverse) FFT, converting half-complex input of size `(nx/2 + 1) × ny` back to a real array of size `nx × ny`. This is the inverse of `DZFFT2D`.

### Interface

```
zdfft2d(a, nx, ny, iopt, b)
```

- **a[(nx/2+1) × ny]**: complex input (half-complex) / real output (`nx × ny`)
- **b[(nx/2+1) × ny]**: work vector
- **nx, ny**: transform lengths (`each = 2^p * 3^q * 5^r`)
- **iopt**: `0` = initialize, `+1` = inverse transform

### Pseudocode

```
function ZDFFT2D(a, da, nx, ny, iopt):
    if iopt == 0:
        SETTBL(wx, nx), SETTBL(wy, ny)
        return

    dn = 1 / (nx · ny)

    // Phase 1: Blocked ny-point inverse FFTs on each frequency bin
    for i = 0 to nx/2 step NBLK:
        extract column conj(a[i][0..ny-1])
        FFT235(column, wy, ny)
        store into b[i][0..ny-1]

    // Phase 2: Reconstruct full spectrum + inverse x-direction FFTs
    for j = 0 to ny-1 step 2:
        // Reconstruct full nx-point complex spectrum from half-complex
        cx[0] = (Re(b[0][j]), Re(b[0][j+1]))
        for i = 1 to nx/2:
            temp = i · b[i][j+1]
            cx[i]    = b[i][j] + temp
            cx[nx-i] = conj(b[i][j] - temp)

        FFT235(cx, wx, nx)

        // Extract two real rows from complex result
        da[·][j]   = Re(cx[·]) · dn
        da[·][j+1] = Im(cx[·]) · dn
```

### Notes

- Phase 1 performs conjugate + forward FFTs along y, which is equivalent to inverse FFTs.
- Phase 2 reconstructs the full `nx`-point spectrum from the half-complex representation, then uses the packed-pair technique in reverse to extract two real output rows from each complex inverse FFT.
- The scaling factor `1/(nx·ny)` is applied to the final real output.
- This is the exact inverse of the `DZFFT2D` (real-to-complex) routine.
- This is the FFTE library's `ZDFFT2D` routine.
