# 3-D Real-to-Complex FFT Routine (with OpenMP and Cache Blocking)

## Algorithm

Computes a 3-D real-to-complex FFT of an `nx × ny × nz` real array. The output exploits **conjugate symmetry**, storing only `(nx/2 + 1) × ny × nz` complex values. Uses a packed-pair technique for the x-direction and cache blocking for all dimensions.

### Interface

```
dzfft3d(a, nx, ny, nz, iopt, b)
```

- **a[nx × ny × nz]**: real input / complex output (half-complex: `(nx/2+1) × ny × nz`)
- **b[(nx/2+1) × ny × nz]**: work vector
- **nx, ny, nz**: transform lengths (`each = 2^p * 3^q * 5^r`)
- **iopt**: `0` = initialize, `-1` = forward transform

### Pseudocode

```
function DZFFT3D(da, a, nx, ny, nz, iopt):
    if iopt == 0:
        SETTBL(wx, nx), SETTBL(wy, ny), SETTBL(wz, nz)
        return

    // Phase 1: Packed real-to-complex x-direction FFTs (per z-slab)
    for k = 0 to nz-1:
        for j = 0 to ny-1 step 2:
            cx[i] = da[i][j][k] + i·da[i][j+1][k]
            FFT235(cx, wx, nx)
            unpack b[0..nx/2][j][k] and b[0..nx/2][j+1][k]

    // Phase 2: Blocked ny-point FFTs (per z-slab)
    for k = 0 to nz-1:
        for i = 0 to nx/2 step NBLK:
            extract column b[i][0..ny-1][k]
            FFT235(column, wy, ny)
            store back

    // Phase 3: Blocked nz-point FFTs (z-direction)
    for j = 0 to ny-1:
        for i = 0 to nx/2 step NBLK:
            extract pencil b[i][j][0..nz-1]
            FFT235(pencil, wz, nz)
            store into a[i][j][0..nz-1]
```

### Notes

- Three phases: x-direction (packed real-to-complex), y-direction (complex FFTs), z-direction (complex FFTs).
- The **packed-pair technique** in Phase 1 processes two real rows simultaneously, halving the number of x-direction FFTs.
- Output uses **half-complex** format: `nx/2 + 1` frequency bins along x.
- Cache blocking (`NBLK`) and padding (`NP`) are applied throughout.
- OpenMP parallelism can be applied at the z-slab level (Phase 1-2) and y-row level (Phase 3).
- This is the FFTE library's `DZFFT3D` routine.
