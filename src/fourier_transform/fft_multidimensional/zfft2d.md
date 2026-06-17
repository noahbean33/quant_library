# 2-D Complex FFT Routine (with OpenMP and Cache Blocking)

## Algorithm

Computes a 2-D complex FFT of size `nx × ny` using a **blocked row-column approach** with OpenMP parallelism. The y-dimension FFTs are performed with cache-blocked transpositions, followed by column-wise x-dimension FFTs.

### Interface

```
zfft2d(a, nx, ny, iopt)
```

- **a[nx × ny]**: complex input/output matrix (row-major)
- **nx, ny**: transform lengths (`each = 2^p * 3^q * 5^r`)
- **iopt**: `0` = initialize, `-1` = forward FFT, `+1` = inverse FFT

### Pseudocode

```
function ZFFT2D(a, nx, ny, iopt):
    if iopt == 0:
        SETTBL(wx, nx)
        SETTBL(wy, ny)
        return

    if iopt == +1:
        conjugate all elements of a

    // OpenMP parallel region:
    ZFFT2D0(a, wx, wy, nx, ny)

    if iopt == +1:
        conjugate and scale a by 1/(nx·ny)
```

### Inner Blocked Computation (ZFFT2D0)

```
function ZFFT2D0(a, wx, wy, nx, ny):
    // Phase 1: Blocked ny-point FFTs (y-dimension)
    for ii = 0 to nx-1 step NBLK:
        // Blocked transpose: extract rows into local buffer
        for i in block [ii, ii+NBLK):
            copy row i to buffer b (length ny)
            FFT235(b, wy, ny)
            copy b back to row i

    // Phase 2: nx-point FFTs (x-dimension, column-wise)
    for j = 0 to ny-1:           // parallelizable
        extract column j (length nx)
        FFT235(column, wx, nx)
        store column j back
```

### Notes

- **Cache blocking** with `NBLK = 16` ensures the transposition and FFT operations stay within L2 cache.
- The y-dimension FFTs use a blocked transpose pattern to convert column access to contiguous access.
- The x-dimension FFTs operate directly on columns, each of which can be processed independently by an OpenMP thread.
- The inverse transform uses the conjugation identity: `IFFT₂D(x) = conj(FFT₂D(conj(x))) / (nx·ny)`.
