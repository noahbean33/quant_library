# Vectorized Radix-2 FFT Kernel Using SSE3 Intrinsics

## Algorithm

An SSE3-vectorized radix-2 butterfly kernel that processes complex data stored as interleaved doubles. Each complex number occupies two consecutive doubles (real, imag), loaded as a single `__m128d`. The butterfly computes sum and twiddle-multiplied difference using the `ZMUL` SSE3 complex multiply.

### Parameters

- `a`: input array of complex values (interleaved doubles)
- `b`: output array of complex values (interleaved doubles)
- `w`: twiddle factor array (interleaved doubles)
- `m`: number of butterflies per group (multirow width)
- `l`: number of groups

### Pseudocode

```
function fft_vec(a, b, w, m, l):
    for j = 0 to l - 1:
        w0 = load_complex(w[j])           // twiddle factor for group j
        for i = 0 to m - 1:
            // Input indices (Stockham addressing)
            i0 = i + j*m                   // first half
            i1 = i0 + m*l                  // second half

            // Output indices (interleaved)
            i2 = i + 2*j*m                 // even output
            i3 = i2 + m                    // odd output

            t0 = load_complex(a[i0])
            t1 = load_complex(a[i1])

            store_complex(b[i2], t0 + t1)
            store_complex(b[i3], ZMUL(w0, t0 - t1))
```

### Notes

- This is the SSE3 implementation of the vectorizable radix-2 kernel — each `_mm_load_pd` / `_mm_store_pd` handles one complex number (2 doubles = 128 bits).
- The `ZMUL` function (SSE3 complex multiply) is used for twiddle factor application, avoiding scalar real/imag splitting.
- Index calculations use bit shifts (`<< 1`, `<< 2`) because each complex number occupies 2 doubles in the interleaved layout.
- The inner loop over `m` rows is the vectorizable dimension — with wider SIMD (AVX), multiple `m` iterations can be fused.
