# Complex Multiplication Using SSE3 Intrinsics

## Algorithm

Multiplies two complex numbers packed in SSE2 128-bit registers using SSE3 instructions. Each `__m128d` holds one complex number as `[real, imag]`. The key SSE3 instruction is `_mm_addsub_pd`, which simultaneously subtracts the low element and adds the high element — perfectly matching the real/imaginary parts of complex multiplication.

### Formula

```
(a.r + i·a.i) × (b.r + i·b.i) = (a.r·b.r - a.i·b.i) + i·(a.r·b.i + a.i·b.r)
```

### Pseudocode (SSE3 register operations)

```
function ZMUL(a, b):
    // a = [a.r, a.i],  b = [b.r, b.i]

    ar = [a.r, a.r]               // movedup: duplicate real part
    ar = [a.r*b.r, a.r*b.i]      // mul ar * b

    ai = [a.i, a.i]               // unpackhi: duplicate imag part
    b  = [b.i, b.r]               // shuffle: swap real and imag of b

    ai = [a.i*b.i, a.i*b.r]      // mul ai * swapped b

    return addsub(ar, ai)         // [a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r]
```

### Notes

- Uses only **5 SSE instructions** (movedup, mul, unpackhi, shuffle, mul, addsub) — no branches or extra storage.
- `_mm_addsub_pd` is the SSE3 instruction that makes this possible: it subtracts the lower double and adds the upper double in a single operation.
- This is the core twiddle-factor multiplication kernel used in vectorized FFT implementations.
- The function is marked `static __inline` for zero call overhead when used in tight FFT loops.
