# Factorization Routine

## Algorithm

Provides two utility routines used throughout the FFTE library:

1. **FACTOR**: Decomposes an integer `n` into its prime factors of 2, 3, and 5.
2. **GETNXNY**: Splits `n` into `nx × ny` where `nx ≈ √n` and both `nx`, `ny` are smooth numbers (products of 2, 3, 5).

### FACTOR

```
function FACTOR(n) → ip[3]:
    // ip[0] = power of 2, ip[1] = power of 3, ip[2] = power of 5
    ip = [0, 0, 0]
    n2 = n
    if n is not divisible by 2, 3, or 5: return

    while n2 > 1:
        if n2 mod 2 == 0:  ip[0]++, n2 /= 2
        else if n2 mod 3 == 0:  ip[1]++, n2 /= 3
        else if n2 mod 5 == 0:  ip[2]++, n2 /= 5
        else: break
```

### GETNXNY

```
function GETNXNY(n) → (nx, ny):
    isqrtn = floor(√n)
    ip = FACTOR(n)

    // Search for nx = 2^a · 3^b · 5^c closest to √n
    best = isqrtn
    for k = 0 to ceil(ip[2]/2):
        for j = 0 to ceil(ip[1]/2):
            for i = 0 to ceil(ip[0]/2):
                candidate = 2^i · 3^j · 5^k
                if candidate ≤ isqrtn and (isqrtn - candidate) < best:
                    lnx = [i, j, k]
                    best = isqrtn - candidate

    nx = 2^lnx[0] · 3^lnx[1] · 5^lnx[2]
    ny = n / nx
```

### Notes

- `FACTOR` is the foundation for determining which radix kernels (2, 3, 4, 5, 8) to apply in the mixed-radix FFT.
- `GETNXNY` is used by the six-step FFT to find a balanced split `n = nx × ny`. A balanced split minimizes the total work and improves cache utilization.
- Only **5-smooth numbers** (regular numbers) are supported: `n = 2^a · 3^b · 5^c`.
- The search in `GETNXNY` is exhaustive but fast, since the number of candidate factorizations is small.
