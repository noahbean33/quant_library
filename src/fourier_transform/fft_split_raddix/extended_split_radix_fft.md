# Extended Split-Radix Decimation-in-Frequency FFT

## Reference

D. Takahashi, "An extended split-radix FFT algorithm," IEEE Signal Processing Lett., vol. 8, pp. 145–147, May 2001.

## Algorithm

Computes the DFT of `n = 2^m` complex values stored in separate real (`X`) and imaginary (`Y`) arrays. Uses a **beta vector** to selectively skip redundant butterflies, reducing the operation count below a standard split-radix FFT.

The `IV` parameter controls direction: `IV = 1` for forward FFT, `IV = -1` for inverse FFT.

### Constants

- `C21 = 1/√2 ≈ 0.70710678118654752`
- `TWOPI = 2π ≈ 6.28318530717958647`

### Pseudocode

```
function ESRFFT(X, Y, IBETA, n, m, iv):

    // --- Step 1: Build beta vector table ---
    IBETA[0] = 1
    IBETA[1] = IBETA[2] = IBETA[3] = 0
    id = 1
    for k = 3 to m-1:
        for j = 1 to 4:
            is = (j + 3) * id
            for i = is to is + id - 1:
                IBETA[i] = IBETA[i - is]
        id = 2 * id

    // --- Step 2: Conjugate imaginary part for inverse FFT ---
    if iv == -1:
        for i = 0 to n-1:
            Y[i] = -Y[i]

    // --- Step 3: L-shaped butterflies (radix-8 stages) ---
    L = 1
    n2 = n
    for k = 1 to m-2:
        n8 = n2 / 8
        e = -2π / n2
        a = 0
        for j = 0 to n8 - 1:
            // Compute twiddle factors for harmonics 1, 3, 5, 7
            cc1 = cos(a);   ss1 = sin(a)
            cc3 = cos(3a);  ss3 = sin(3a)
            cc5 = cos(5a);  ss5 = sin(5a)
            cc7 = cos(7a);  ss7 = sin(7a)
            a = (j + 1) * e

            for i = 0 to L - 1:
                if IBETA[i] == 1:
                    // Compute 8 indices spaced n8 apart
                    i0 = i * n2 + j
                    i1 = i0 + n8;  i2 = i1 + n8;  i3 = i2 + n8
                    i4 = i3 + n8;  i5 = i4 + n8;  i6 = i5 + n8;  i7 = i6 + n8

                    // Differences (upper - lower halves)
                    x0 = X[i0] - X[i4];    y0 = Y[i0] - Y[i4]
                    x1 = X[i1] - X[i5];    y1 = Y[i1] - Y[i5]
                    x2 = Y[i2] - Y[i6];    y2 = X[i6] - X[i2]   // note: j-rotation
                    x3 = X[i3] - X[i7];    y3 = Y[i3] - Y[i7]

                    // Sums (in-place update of upper half)
                    X[i0] += X[i4];  Y[i0] += Y[i4]
                    X[i1] += X[i5];  Y[i1] += Y[i5]
                    X[i2] += X[i6];  Y[i2] += Y[i6]
                    X[i3] += X[i7];  Y[i3] += Y[i7]

                    // Combine with √2/2 weighting
                    u0 = x0 + C21*(x1 - x3);  v0 = y0 + C21*(y1 - y3)
                    u1 = x0 - C21*(x1 - x3);  v1 = y0 - C21*(y1 - y3)
                    u2 = x2 + C21*(y1 + y3);  v2 = y2 - C21*(x1 + x3)
                    u3 = x2 - C21*(y1 + y3);  v3 = y2 + C21*(x1 + x3)

                    // Apply twiddle factors to lower half
                    X[i4] = cc1*(u0+u2) - ss1*(v0+v2)
                    Y[i4] = cc1*(v0+v2) + ss1*(u0+u2)
                    X[i5] = cc5*(u1+u3) - ss5*(v1+v3)
                    Y[i5] = cc5*(v1+v3) + ss5*(u1+u3)
                    X[i6] = cc3*(u1-u3) - ss3*(v1-v3)
                    Y[i6] = cc3*(v1-v3) + ss3*(u1-u3)
                    X[i7] = cc7*(u0-u2) - ss7*(v0-v2)
                    Y[i7] = cc7*(v0-v2) + ss7*(u0-u2)

        L = 2 * L
        n2 = n2 / 2

    // --- Step 4: Length-4 butterflies ---
    for i = 0 to n/4 - 1:
        if IBETA[i] == 1:
            i0 = 4*i;  i1 = i0+1;  i2 = i1+1;  i3 = i2+1
            x0 = X[i0] - X[i2];     y0 = Y[i0] - Y[i2]
            x1 = Y[i1] - Y[i3];     y1 = X[i3] - X[i1]
            X[i0] += X[i2];  Y[i0] += Y[i2]
            X[i1] += X[i3];  Y[i1] += Y[i3]
            X[i2] = x0 + x1;  Y[i2] = y0 + y1
            X[i3] = x0 - x1;  Y[i3] = y0 - y1

    // --- Step 5: Length-2 butterflies ---
    for i = 0 to n/2 - 1:
        if IBETA[i] == 1:
            i0 = 2*i;  i1 = i0+1
            x0 = X[i0] - X[i1];  y0 = Y[i0] - Y[i1]
            X[i0] += X[i1];  Y[i0] += Y[i1]
            X[i1] = x0;  Y[i1] = y0

    // --- Step 6: Bit-reversal permutation ---
    j = 0
    for i = 0 to n-2:
        if i < j:
            swap(X[i], X[j])
            swap(Y[i], Y[j])
        k = n / 2
        while k <= j:
            j = j - k
            k = k / 2
        j = j + k

    // --- Step 7: Normalize for inverse FFT ---
    if iv == -1:
        for i = 0 to n-1:
            X[i] = X[i] / n
            Y[i] = -Y[i] / n
```

### Notes

- The **beta vector** (`IBETA`) marks which butterfly groups need computation, skipping redundant operations. This is the key innovation of the extended split-radix approach.
- The L-shaped butterfly stage processes 8 points at a time using twiddle factors at harmonics 1, 3, 5, and 7 of the base angle.
- After the radix-8 stages, residual radix-4 and radix-2 passes complete the transform.
- A standard **bit-reversal permutation** reorders the output.
- For the **inverse FFT** (`iv = -1`), the imaginary part is negated before the transform and the output is conjugated and scaled by `1/n`.
