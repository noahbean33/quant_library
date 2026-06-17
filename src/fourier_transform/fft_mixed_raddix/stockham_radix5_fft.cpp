// Radix-5 FFT Algorithm (Stockham auto-sort)

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

using Complex = std::complex<double>;

static const double PI        = 4.0 * std::atan(1.0);
static const double SIN_2PI_5 = std::sin(2.0 * PI / 5.0);
static const double SIN_PI_5  = std::sin(PI / 5.0);
static const double SQRT5_4   = std::sqrt(5.0) / 4.0;
static const double RATIO     = SIN_PI_5 / SIN_2PI_5;  // sin(π/5) / sin(2π/5)

void radix5_fft(std::vector<Complex>& x, int n) {
    // Compute p where n = 5^p
    int p = 0;
    for (int tmp = n; tmp > 1; tmp /= 5) {
        ++p;
    }

    std::vector<Complex> y(n);
    std::vector<Complex>* X_prev = &x;
    std::vector<Complex>* X_next = &y;

    int l = n / 5;
    int m = 1;

    for (int t = 0; t < p; ++t) {
        double angle_base = -2.0 * PI / static_cast<double>(5 * l);

        for (int j = 0; j < l; ++j) {
            Complex w1(std::cos(angle_base * j), std::sin(angle_base * j));
            Complex w2 = w1 * w1;
            Complex w3 = w2 * w1;
            Complex w4 = w3 * w1;

            for (int k = 0; k < m; ++k) {
                Complex c0 = (*X_prev)[k + j * m];
                Complex c1 = (*X_prev)[k + j * m + l * m];
                Complex c2 = (*X_prev)[k + j * m + 2 * l * m];
                Complex c3 = (*X_prev)[k + j * m + 3 * l * m];
                Complex c4 = (*X_prev)[k + j * m + 4 * l * m];

                Complex d0 = c1 + c4;
                Complex d1 = c2 + c3;
                Complex d2 = SIN_2PI_5 * (c1 - c4);
                Complex d3 = SIN_2PI_5 * (c2 - c3);
                Complex d4 = d0 + d1;
                Complex d5 = SQRT5_4 * (d0 - d1);
                Complex d6 = c0 - 0.25 * d4;
                Complex d7 = d6 + d5;
                Complex d8 = d6 - d5;
                Complex d9  = Complex(0.0, -1.0) * (d2 + RATIO * d3);
                Complex d10 = Complex(0.0, -1.0) * (RATIO * d2 - d3);

                (*X_next)[k + 5 * j * m]         = c0 + d4;
                (*X_next)[k + 5 * j * m + m]     = w1 * (d7 + d9);
                (*X_next)[k + 5 * j * m + 2 * m] = w2 * (d8 + d10);
                (*X_next)[k + 5 * j * m + 3 * m] = w3 * (d8 - d10);
                (*X_next)[k + 5 * j * m + 4 * m] = w4 * (d7 - d9);
            }
        }

        l /= 5;
        m *= 5;
        std::swap(X_prev, X_next);
    }

    if (X_prev == &y) {
        x = y;
    }
}
