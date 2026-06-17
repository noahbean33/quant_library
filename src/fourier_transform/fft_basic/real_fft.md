# Complex-to-Real FFT Routine

## Algorithm

Given a real-valued input array `a` of `n` elements (stored as `n/2 + 1` complex pairs), compute the real DFT by first performing an `n/2`-point complex FFT and then unscrambling the result using symmetry properties of real-valued signals.

### Pseudocode

```
function RealFFT(a, n):
    // a is stored as a 2D array: a[2][n/2+1]
    //   a[0][i] = real part, a[1][i] = imaginary part

    // Step 1: Perform n/2-point complex FFT on packed data
    ComplexFFT(a, n/2)

    // Step 2: Unscramble DC and Nyquist components
    temp      = a[0][0] - a[1][0]
    a[0][0]   = a[0][0] + a[1][0]
    a[1][0]   = 0
    a[1][n/4] = -a[1][n/4]
    a[0][n/2] = temp
    a[1][n/2] = 0

    // Step 3: Unscramble remaining frequency bins using conjugate symmetry
    px = -2π / n
    for i = 1 to n/4 - 1:
        w  = px * i
        ar = 0.5 * (a[0][i] - a[0][n/2 - i])
        ai = 0.5 * (a[1][i] + a[1][n/2 - i])
        wr = 1.0 - sin(w)
        wi = cos(w)
        temp      = ar * wi + ai * wr
        ar        = ar * wr - ai * wi
        ai        = temp
        a[0][i]         = a[0][i]         - ar
        a[1][i]         = a[1][i]         - ai
        a[0][n/2 - i]   = a[0][n/2 - i]   + ar
        a[1][n/2 - i]   = a[1][n/2 - i]   - ai
```

### Notes

- This exploits the **conjugate symmetry** of the DFT of real-valued data: `X[k] = conj(X[n-k])`.
- A real-valued `n`-point FFT is computed using only an `n/2`-point complex FFT, halving the work.
- The unscrambling step separates the even and odd DFT components that were interleaved by packing the real data into a complex array.
- The twiddle factor `(wr, wi) = (1 - sin(w), cos(w))` comes from the half-sample shift correction.
