# 3-D Complex FFT Routine (with OpenMP and Cache Blocking)

## Algorithm

Computes a 3-D complex FFT of size `nx × ny × nz` using a **blocked row-column approach** with OpenMP parallelism. Each dimension is transformed using mixed-radix FFTs with cache-blocked data access.

### Interface

```
zfft3d(a, nx, ny, nz, iopt)
```

- **a[nx × ny × nz]**: complex input/output array
- **nx, ny, nz**: transform lengths (`each = 2^p * 3^q * 5^r`)
- **iopt**: `0` = initialize, `-1` = forward FFT, `+1` = inverse FFT

### Pseudocode

```
function ZFFT3D(a, nx, ny, nz, iopt):
    if iopt == 0:
        SETTBL(wx, nx), SETTBL(wy, ny), SETTBL(wz, nz)
        return

    if iopt == +1:
        conjugate all elements of a

    // OpenMP parallel region:
    ZFFT3D0(a, wx, wy, wz, nx, ny, nz)

    if iopt == +1:
        conjugate and scale a by 1/(nx·ny·nz)
```

### Inner Blocked Computation (ZFFT3D0)

```
function ZFFT3D0(a, wx, wy, wz, nx, ny, nz):
    // Phase 1: nz-point FFTs along z-dimension
    for j = 0 to ny-1:                     // parallelizable
        for ii = 0 to nx-1 step NBLK:
            for i in block:
                extract z-pencil a[i][j][0..nz-1]
                FFT235(pencil, wz, nz)
                store back

    // Phase 2: ny-point FFTs along y-dimension
    for k = 0 to nz-1:                     // parallelizable
        for ii = 0 to nx-1 step NBLK:
            for i in block:
                extract y-pencil a[i][0..ny-1][k]
                FFT235(pencil, wy, ny)
                store back

    // Phase 3: nx-point FFTs along x-dimension
        for j = 0 to ny-1:
            extract x-pencil a[0..nx-1][j][k]
            FFT235(pencil, wx, nx)
            store back
```

### Notes

- Three phases transform each dimension (z → y → x), with **cache blocking** (`NBLK = 16`) applied to the blocked extraction of pencils.
- **Padding** (`NP = 8`) is added to work arrays to avoid cache-line conflicts.
- OpenMP parallelism is applied at the outermost loop of each phase.
- The inverse transform uses the conjugation identity: `IFFT₃D(x) = conj(FFT₃D(conj(x))) / (nx·ny·nz)`.
- This routine corresponds to the FFTE library's `ZFFT3D` subroutine.
