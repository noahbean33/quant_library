// Vectorizable radix-2 FFT kernel

#include <cmath>
#include <complex>
#include <vector>

using Complex = std::complex<double>;

// Radix-2 butterfly kernel with m independent rows (vectorizable).
// a: input  — a[i + m*(j + l*half)], shape (m, l, 2)
// b: output — b[i + m*(half + 2*j)], shape (m, 2, l)
// w: twiddle factors, length l
// m: multirow width (vectorizable dimension)
// l: number of butterfly groups
void vectorizable_radix2_kernel(const std::vector<Complex>& a,
                                std::vector<Complex>& b,
                                const std::vector<Complex>& w,
                                int m, int l) {
    for (int j = 0; j < l; ++j) {
        for (int i = 0; i < m; ++i) {  // vectorizable loop
            Complex c0 = a[i + m * j];
            Complex c1 = a[i + m * j + m * l];
            b[i + m * (2 * j)]     = c0 + c1;
            b[i + m * (2 * j + 1)] = w[j] * (c0 - c1);
        }
    }
}
