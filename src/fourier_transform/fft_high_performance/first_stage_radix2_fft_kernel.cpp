// First stage of a vectorizable radix-2 FFT kernel (m = 1 specialization)

#include <cmath>
#include <complex>
#include <vector>

using Complex = std::complex<double>;

// First-stage radix-2 butterfly kernel (m = 1).
// a: input  — a[j + l*half], shape (l, 2)
// b: output — b[half + 2*j],  shape (2, l)
// w: twiddle factors, length l
// l: number of butterfly groups (= n/2)
void first_stage_radix2_kernel(const std::vector<Complex>& a,
                               std::vector<Complex>& b,
                               const std::vector<Complex>& w,
                               int l) {
    for (int j = 0; j < l; ++j) {
        Complex c0 = a[j];
        Complex c1 = a[j + l];
        b[2 * j]     = c0 + c1;
        b[2 * j + 1] = w[j] * (c0 - c1);
    }
}
