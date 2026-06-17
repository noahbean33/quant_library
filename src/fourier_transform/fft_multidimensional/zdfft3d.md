# 3-D Complex-to-Real FFT Routine (with OpenMP and Cache Blocking)

## Algorithm

Computes a 3-D complex-to-real (inverse) FFT, converting half-complex input of size `(nx/2 + 1) × ny × nz` back to a real array of size `nx × ny × nz`. This is the inverse of `DZFFT3D`.

### Interface

```
zdfft3d(a, nx, ny, nz, iopt, b)
```

- **a[(nx/2+1) × ny × nz]**: complex input (half-complex) / real output (`nx × ny × nz`)
- **b[(nx/2+1) × ny × nz]**: work vector
- **nx, ny, nz**: transform lengths (`each = 2^p * 3^q * 5^r`)
- **iopt**: `0` = initialize, `+1` = inverse transform

### Pseudocode

```
function ZDFFT3D(a, da, nx, ny, nz, iopt):
    if iopt == 0:
        SETTBL(wx, nx), SETTBL(wy, ny), SETTBL(wz, nz)
        return

    dn = 1 / (nx · ny · nz)

    // Phase 1: Blocked nz-point inverse FFTs along z-dimension
    for j = 0 to ny-1:
        for i = 0 to nx/2 step NBLK:
            extract pencil conj(a[i][j][0..nz-1])
            FFT235(pencil, wz, nz)
            store into b[i][j][0..nz-1]

    // Phase 2: Blocked ny-point inverse FFTs (per z-slab)
    for k = 0 to nz-1:
        for i = 0 to nx/2 step NBLK:
            extract column b[i][0..ny-1][k]
            FFT235(column, wy, ny)
            store back

    // Phase 3: Reconstruct full spectrum + inverse x-direction FFTs
    for k = 0 to nz-1:
        for j = 0 to ny-1 step 2:
            cx[0] = (Re(b[0][j][k]), Re(b[0][j+1][k]))
            for i = 1 to nx/2:
                temp = i · b[i][j+1][k]
                cx[i]    = b[i][j][k] + temp
                cx[nx-i] = conj(b[i][j][k] - temp)
            FFT235(cx, wx, nx)
            da[·][j][k]   = Re(cx[·]) · dn
            da[·][j+1][k] = Im(cx[·]) · dn
```

### Notes

- Three phases in reverse order compared to `DZFFT3D`: z-direction → y-direction → x-direction (reconstruct + inverse).
- Phase 1 uses conjugate + forward FFTs along z, equivalent to inverse FFTs.
- Phase 3 reconstructs the full `nx`-point spectrum from half-complex and uses the packed-pair technique in reverse.
- The scaling factor `1/(nx·ny·nz)` is applied to the final real output.
- This is the exact inverse of the `DZFFT3D` (real-to-complex) routine.
- This is the FFTE library's `ZDFFT3D` routine.
