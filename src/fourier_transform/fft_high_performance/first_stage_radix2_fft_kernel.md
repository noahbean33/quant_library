# First Stage of a Vectorizable Radix-2 FFT Kernel

## Algorithm

The specialized first-stage radix-2 butterfly kernel where `m = 1` (single row). This is the initial decomposition step of the Stockham FFT where the input is viewed as `a(l, 2)` — `l` groups with 2 elements each — and the output is `b(2, l)`. Since `m = 1`, there is no inner multirow loop; the vectorizable loop is over the `l` groups instead.

### Parameters

- `a(l, 2)`: input array — `l` groups, 2 elements per butterfly
- `b(2, l)`: output array — 2 outputs per group, `l` groups
- `w(l)`: twiddle factors, one per group
- `l`: number of butterfly groups (= n/2)

### Pseudocode

```
function FirstStageRadix2Kernel(a, b, w, l):
    for j = 0 to l - 1:
        c0 = a[j][0]
        c1 = a[j][1]
        b[0][j] = c0 + c1
        b[1][j] = w[j] * (c0 - c1)
```

### Notes

- This is a specialization of the general vectorizable radix-2 kernel with `m = 1`.
- At the first FFT stage, the stride between butterfly pairs is maximal (`l = n/2`), so each butterfly reads two elements separated by `n/2`.
- The loop over `j` is vectorizable — each iteration is independent.
- Separating the first stage from the general kernel allows the compiler to optimize for the `m = 1` case (no inner loop overhead, simpler addressing).
- In practice, the first stage often has different cache behavior since the input is accessed with large stride.
