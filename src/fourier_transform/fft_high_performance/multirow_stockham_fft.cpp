// Multirow FFT based on Stockham FFT algorithm

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

using Complex = std::complex<double>;

static const double PI = 4.0 * std::atan(1.0);

// Computes FFT of ns independent rows of length n each.
// x: ns x n matrix (row-major: x[row * n + col])
void multirow_stockham_fft(std::vector<Complex>& x, int ns, int n) {
    // Compute p where n = 2^p
    int p = 0;
    for (int tmp = n; tmp > 1; tmp >>= 1) {
        ++p;
    }

    std::vector<Complex> y(static_cast<size_t>(ns) * n);
    std::vector<Complex>* X_prev = &x;
    std::vector<Complex>* X_next = &y;

    int l = n / 2;
    int m = 1;

    for (int t = 0; t < p; ++t) {
        double angle_base = -2.0 * PI / static_cast<double>(2 * l);

        for (int j = 0; j < l; ++j) {
            Complex twiddle(std::cos(angle_base * j), std::sin(angle_base * j));

            for (int k = 0; k < m; ++k) {
                int src0 = k + j * m;
                int src1 = k + j * m + l * m;
                int dst0 = k + 2 * j * m;
                int dst1 = k + 2 * j * m + m;

                // Vectorizable loop over independent rows
                for (int row = 0; row < ns; ++row) {
                    Complex c0 = (*X_prev)[row * n + src0];
                    Complex c1 = (*X_prev)[row * n + src1];
                    (*X_next)[row * n + dst0] = c0 + c1;
                    (*X_next)[row * n + dst1] = twiddle * (c0 - c1);
                }
            }
        }

        l /= 2;
        m *= 2;
        std::swap(X_prev, X_next);
    }

    // If result ended up in y, copy back to x
    if (X_prev == &y) {
        x = y;
    }
}
