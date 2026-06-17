// Radix-4 FFT Algorithm (Stockham auto-sort)

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

using Complex = std::complex<double>;

static const double PI = 4.0 * std::atan(1.0);

void radix4_fft(std::vector<Complex>& x, int n) {
    // Compute p where n = 4^p
    int p = 0;
    for (int tmp = n; tmp > 1; tmp /= 4) {
        ++p;
    }

    std::vector<Complex> y(n);
    std::vector<Complex>* X_prev = &x;
    std::vector<Complex>* X_next = &y;

    int l = n / 4;
    int m = 1;

    for (int t = 0; t < p; ++t) {
        double angle_base = -2.0 * PI / static_cast<double>(4 * l);

        for (int j = 0; j < l; ++j) {
            Complex w1(std::cos(angle_base * j), std::sin(angle_base * j));
            Complex w2 = w1 * w1;
            Complex w3 = w2 * w1;

            for (int k = 0; k < m; ++k) {
                Complex c0 = (*X_prev)[k + j * m];
                Complex c1 = (*X_prev)[k + j * m + l * m];
                Complex c2 = (*X_prev)[k + j * m + 2 * l * m];
                Complex c3 = (*X_prev)[k + j * m + 3 * l * m];

                Complex d0 = c0 + c2;
                Complex d1 = c0 - c2;
                Complex d2 = c1 + c3;
                Complex d3 = Complex(0.0, -1.0) * (c1 - c3);

                (*X_next)[k + 4 * j * m]         = d0 + d2;
                (*X_next)[k + 4 * j * m + m]     = w1 * (d1 + d3);
                (*X_next)[k + 4 * j * m + 2 * m] = w2 * (d0 - d2);
                (*X_next)[k + 4 * j * m + 3 * m] = w3 * (d1 - d3);
            }
        }

        l /= 4;
        m *= 4;
        std::swap(X_prev, X_next);
    }

    if (X_prev == &y) {
        x = y;
    }
}
