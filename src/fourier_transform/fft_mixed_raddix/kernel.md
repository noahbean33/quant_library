# Radix-2, 3, 4, 5 and 8 FFT Kernel Routines

## Algorithm

Low-level butterfly kernel routines for the FFTE mixed-radix FFT. Each radix has two variants:
- **A variant** (`FFTrA`): Optimized for stride `m == 1` (first stage of the FFT).
- **B variant** (`FFTrB`): General stride `m > 1` (subsequent stages).

### Constants

| Symbol | Value | Meaning |
|--------|-------|---------|
| `C31` | 0.86602540378... | `sin(π/3) = √3/2` |
| `C32` | 0.5 | `cos(π/3) = 1/2` |
| `C51` | 0.95105651629... | `sin(2π/5)` |
| `C52` | 0.61803398874... | `2·sin(π/5)·cos(π/5) / sin(2π/5)` |
| `C53` | 0.55901699437... | `(√5 - 1) / 4` |
| `C54` | 0.25 | `1/4` |
| `C81` | 0.70710678118... | `1/√2 = cos(π/4) = sin(π/4)` |

### Radix-2 Butterfly (FFT2)

```
for i = 0 to m-1:
    b[i]     = a[i] + a[i + m]
    b[i + m] = a[i] - a[i + m]
```

### Radix-3 Butterfly (FFT3)

```
for each group j (with twiddle ω = w[j]):
    c0, c1, c2 = input triplet
    d0 = c1 + c2
    d1 = c0 - ½·d0
    d2 = -i·sin(π/3)·(c1 - c2)
    out[0] = c0 + d0
    out[1] = ω   · (d1 + d2)
    out[2] = ω²  · (d1 - d2)
```

### Radix-4 Butterfly (FFT4)

```
for each group j (with twiddles ω₁, ω₂, ω₃):
    c0, c1, c2, c3 = input quartet
    d0 = c0 + c2,  d1 = c0 - c2
    d2 = c1 + c3,  d3 = -i·(c1 - c3)
    out[0] = d0 + d2
    out[1] = ω₁ · (d1 + d3)
    out[2] = ω₂ · (d0 - d2)
    out[3] = ω₃ · (d1 - d3)
```

### Radix-5 Butterfly (FFT5)

```
for each group j (with twiddles ω₁..ω₄):
    c0..c4 = input quintet
    d0 = c1+c4, d1 = c2+c3
    d2 = C51·(c1-c4), d3 = C51·(c2-c3)
    d4 = d0+d1
    d5 = C53·(d0-d1)
    d6 = c0 - C54·d4
    d7 = d6+d5, d8 = d6-d5
    d9  = -i·(d2 + C52·d3)
    d10 = -i·(C52·d2 - d3)
    out[0] = c0 + d4
    out[1] = ω₁·(d7+d9)
    out[2] = ω₂·(d8+d10)
    out[3] = ω₃·(d8-d10)
    out[4] = ω₄·(d7-d9)
```

### Radix-8 Butterfly (FFT8)

```
for each group j (with twiddles ω₁..ω₇):
    c0..c7 = input octet
    // Two radix-4 sub-butterflies + radix-2 combination
    d0 = c0+c4, d1 = c0-c4
    d2 = c2+c6, d3 = -i·(c2-c6)
    d4 = c1+c5, d5 = c1-c5
    d6 = c3+c7, d7 = c3-c7
    e0 = d0+d2, e1 = d0-d2
    e2 = d4+d6, e3 = -i·(d4-d6)
    e4 = (1/√2)·(d5-d7)
    e5 = -i·(1/√2)·(d5+d7)
    e6 = d1+e4, e7 = d1-e4
    e8 = d3+e5, e9 = d3-e5
    out[0] = e0+e2
    out[1] = ω₁·(e6+e8)
    out[2] = ω₂·(e1+e3)
    out[3] = ω₃·(e7-e9)
    out[4] = ω₄·(e0-e2)
    out[5] = ω₅·(e7+e9)
    out[6] = ω₆·(e1-e3)
    out[7] = ω₇·(e6-e8)
```

### Notes

- Each butterfly reads `r` inputs and produces `r` outputs, performing the length-`r` DFT with twiddle factor multiplication.
- The **A variants** (stride 1) have the inner loop over `j` (groups), while the **B variants** have the inner loop over `i` (elements within a group).
- In the B variants, the first group (`j == 0`) skips twiddle multiplication since `ω^0 = 1`.
- The radix-8 kernel is decomposed into two radix-4 sub-butterflies combined with radix-2 operations, using the `1/√2` factor for the 45° rotations.
- These are the computational core of the FFTE library, and are designed for vectorization by the compiler.
