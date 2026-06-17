// Radix-8 FFT Algorithm (Stockham auto-sort)

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

using Complex = std::complex<double>;

static const double PI      = 4.0 * std::atan(1.0);
static const double SQRT2_2 = std::sqrt(2.0) / 2.0;  // √2/2

void radix8_fft(std::vector<Complex>& x, int n) {
    // Compute p where n = 8^p
    int p = 0;
    for (int tmp = n; tmp > 1; tmp /= 8) {
        ++p;
    }

    std::vector<Complex> y(n);
    std::vector<Complex>* X_prev = &x;
    std::vector<Complex>* X_next = &y;

    int l = n / 8;
    int m = 1;

    for (int t = 0; t < p; ++t) {
        double angle_base = -2.0 * PI / static_cast<double>(8 * l);

        for (int j = 0; j < l; ++j) {
            Complex w1(std::cos(angle_base * j), std::sin(angle_base * j));
            Complex w2 = w1 * w1;
            Complex w3 = w2 * w1;
            Complex w4 = w3 * w1;
            Complex w5 = w4 * w1;
            Complex w6 = w5 * w1;
            Complex w7 = w6 * w1;

            for (int k = 0; k < m; ++k) {
                Complex c0 = (*X_prev)[k + j * m];
                Complex c1 = (*X_prev)[k + j * m + l * m];
                Complex c2 = (*X_prev)[k + j * m + 2 * l * m];
                Complex c3 = (*X_prev)[k + j * m + 3 * l * m];
                Complex c4 = (*X_prev)[k + j * m + 4 * l * m];
                Complex c5 = (*X_prev)[k + j * m + 5 * l * m];
                Complex c6 = (*X_prev)[k + j * m + 6 * l * m];
                Complex c7 = (*X_prev)[k + j * m + 7 * l * m];

                // Stage 1: radix-2 pairs
                Complex d0 = c0 + c4;
                Complex d1 = c0 - c4;
                Complex d2 = c2 + c6;
                Complex d3 = Complex(0.0, -1.0) * (c2 - c6);
                Complex d4 = c1 + c5;
                Complex d5 = c1 - c5;
                Complex d6 = c3 + c7;
                Complex d7 = c3 - c7;

                // Stage 2: radix-4 combinations
                Complex e0 = d0 + d2;
                Complex e1 = d0 - d2;
                Complex e2 = d4 + d6;
                Complex e3 = Complex(0.0, -1.0) * (d4 - d6);
                Complex e4 = SQRT2_2 * (d5 - d7);
                Complex e5 = Complex(0.0, -SQRT2_2) * (d5 + d7);
                Complex e6 = d1 + e4;
                Complex e7 = d1 - e4;
                Complex e8 = d3 + e5;
                Complex e9 = d3 - e5;

                // Stage 3: output with twiddle factors
                (*X_next)[k + 8 * j * m]         = e0 + e2;
                (*X_next)[k + 8 * j * m + m]     = w1 * (e6 + e8);
                (*X_next)[k + 8 * j * m + 2 * m] = w2 * (e1 + e3);
                (*X_next)[k + 8 * j * m + 3 * m] = w3 * (e7 - e9);
                (*X_next)[k + 8 * j * m + 4 * m] = w4 * (e0 - e2);
                (*X_next)[k + 8 * j * m + 5 * m] = w5 * (e7 + e9);
                (*X_next)[k + 8 * j * m + 6 * m] = w6 * (e1 - e3);
                (*X_next)[k + 8 * j * m + 7 * m] = w7 * (e6 - e8);
            }
        }

        l /= 8;
        m *= 8;
        std::swap(X_prev, X_next);
    }

    if (X_prev == &y) {
        x = y;
    }
}
