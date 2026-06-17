// Stockham FFT algorithm

#include <cmath>
#include <complex>
#include <vector>

using Complex = std::complex<double>;

void stockham_fft(std::vector<Complex>& x, int n) {
    const double pi = 4.0 * std::atan(1.0);

    // Compute p where n = 2^p
    int p = 0;
    for (int tmp = n; tmp > 1; tmp >>= 1) {
        ++p;
    }

    std::vector<Complex> y(n);

    std::vector<Complex>* X_prev = &x;
    std::vector<Complex>* X_next = &y;

    int l = n / 2;
    int m = 1;

    for (int t = 0; t < p; ++t) {
        double angle_base = -2.0 * pi / static_cast<double>(2 * l);

        for (int j = 0; j < l; ++j) {
            Complex twiddle(std::cos(angle_base * j), std::sin(angle_base * j));
            for (int k = 0; k < m; ++k) {
                Complex c0 = (*X_prev)[k + j * m];
                Complex c1 = (*X_prev)[k + j * m + l * m];
                (*X_next)[k + 2 * j * m]     = c0 + c1;
                (*X_next)[k + 2 * j * m + m] = twiddle * (c0 - c1);
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
