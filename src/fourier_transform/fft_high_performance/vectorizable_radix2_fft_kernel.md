# Vectorizable Radix-2 FFT Kernel

## Algorithm

A radix-2 butterfly kernel designed for vectorization across `m` rows. The input is viewed as a 3-D array `a(m, l, 2)` — `m` simultaneous butterflies across `l` groups with 2 elements each. The output `b(m, 2, l)` has the interleaved Stockham layout. The inner loop over `m` rows has no data dependencies, making it auto-vectorizable by the compiler.

### Parameters

- `a(m, l, 2)`: input array — `m` rows, `l` groups, 2 elements per butterfly
- `b(m, 2, l)`: output array — `m` rows, 2 outputs per group, `l` groups
- `w(l)`: twiddle factors, one per group
- `m`: multirow width (number of independent butterflies)
- `l`: number of butterfly groups

### Pseudocode

```
function VectorizableRadix2Kernel(a, b, w, m, l):
    for j = 0 to l - 1:
        for i = 0 to m - 1:           // vectorizable loop
            c0 = a[i][j][0]
            c1 = a[i][j][1]
            b[i][0][j] = c0 + c1
            b[i][1][j] = w[j] * (c0 - c1)
```

### Notes

- The inner `i` loop iterates over `m` independent rows — no cross-iteration dependencies — so the compiler can vectorize it with SIMD instructions.
- This is the **general stage** kernel: `m ≥ 1` rows are processed simultaneously, with `l` groups and stride determined by the FFT decomposition level.
- The twiddle factor `w[j]` is constant across all `m` rows in each group, enabling broadcast-style SIMD multiplication.
- This Fortran-style kernel maps directly to the SSE3 intrinsics version (`sse3_radix2_fft_kernel`), where each complex value becomes one `__m128d`.
