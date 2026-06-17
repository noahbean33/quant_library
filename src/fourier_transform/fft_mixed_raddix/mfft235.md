# Radix-2, 3, 4, 5 and 8 Multiple FFT and Transposition Routines

## Algorithm

Provides multi-column FFT routines and matrix transposition utilities used by the higher-level 1-D, 2-D, and 3-D FFT routines in the FFTE library.

### MFFT235A: Multiple FFT (Variant A)

Performs `ns` simultaneous `n`-point FFTs. The data is laid out as `ns` interleaved columns of length `n`. Uses FFTrB (general stride) kernels throughout.

```
function MFFT235A(a, b, w, ns, n, ip):
    // Same stage decomposition as FFT235, but with stride ns*m
    // Result ends up in a
    for each radix stage:
        FFTrB(src, dst, w[j:], ns*m, l)
        swap src/dst based on key
```

### MFFT235B: Multiple FFT (Variant B)

Similar to MFFT235A but the final result ends up in `b` (not `a`). Uses mixed A/B dispatch. The last-stage handling differs to ensure correct output placement.

```
function MFFT235B(a, b, w, ns, n, ip):
    // Same structure as MFFT235A
    // Result ends up in b
```

### ZTRANS: Matrix Transposition

Transposes an `nx × ny` complex matrix to `ny × nx`. Two implementations:

- **ZTRANSA**: Simple row-by-row transpose. Used when neither dimension is a power of 2 (avoids cache conflicts naturally).
- **ZTRANSB**: Diagonal traversal transpose. Used when dimensions are powers of 2, where the diagonal access pattern avoids cache-set conflicts.

```
function ZTRANS(a, b, nx, ny):
    if nx == 1 or ny == 1:
        copy a → b
    else if power-of-2 count ≤ 1:
        ZTRANSA(a, b, nx, ny)    // simple transpose
    else:
        ZTRANSB(a, b, nx, ny)    // diagonal transpose
```

### ZTRANS2: Batched 2-D Transpose

Transposes each slab of a 3-D array independently:

```
function ZTRANS2(a, b, nx, ny, nz):
    for k = 0 to nz-1:
        ZTRANS(a[:,:,k], b[:,:,k], nx, ny)
```

### MZTRANS: Multi-Stride Transpose

Transposes with a leading batch dimension `ns`:

```
function MZTRANS(a, b, ns, nx, ny):
    if ns == 1:
        ZTRANS(a, b, nx, ny)
    else:
        for i, j, k:
            b[k + j·ns + i·ns·ny] = a[k + i·ns + j·ns·nx]
```

### Notes

- **MFFT235A** leaves the result in the input array `a`; **MFFT235B** leaves it in `b`. The choice depends on whether a subsequent transposition step expects the data in the source or destination buffer.
- The diagonal traversal in **ZTRANSB** is critical for power-of-2 dimensions, where naive row-by-row transposition causes severe cache-set conflicts (all rows map to the same cache lines).
- The selection between ZTRANSA and ZTRANSB is based on the sum of power-of-2 exponents in `nx` and `ny`: if at most one dimension is a power of 2, simple transpose suffices.
- These routines are the building blocks for the six-step FFT and multi-dimensional FFT algorithms.
