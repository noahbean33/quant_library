// Decimation-in-frequency FFT routine (recursive)

#include <cmath>
#include <complex>
#include <vector>

using Complex = std::complex<double>;

void fft(std::vector<Complex>& x, std::vector<Complex>& temp, int n) {
    if (n <= 1) return;

    const double pi = 4.0 * std::atan(1.0);
    const double px = -2.0 * pi / static_cast<double>(n);

    // Butterfly operations
    for (int j = 0; j < n / 2; ++j) {
        double w = px * static_cast<double>(j);
        temp[j]         = x[j] + x[j + n / 2];
        temp[j + n / 2] = (x[j] - x[j + n / 2]) * Complex(std::cos(w), std::sin(w));
    }

    // Recurse on each half (using x as temporary workspace)
    std::vector<Complex> x_half(x.begin(), x.begin() + n);
    fft(temp, x_half, n / 2);

    std::vector<Complex> temp_upper(temp.begin() + n / 2, temp.begin() + n);
    std::vector<Complex> x_upper(n / 2);
    fft(temp_upper, x_upper, n / 2);
    for (int j = 0; j < n / 2; ++j) {
        temp[j + n / 2] = temp_upper[j];
    }

    // Interleave results back into x
    for (int j = 0; j < n / 2; ++j) {
        x[2 * j]     = temp[j];
        x[2 * j + 1] = temp[j + n / 2];
    }
}
