// Cooley-Tukey FFT algorithm (iterative, in-place)

#include <cmath>
#include <complex>
#include <vector>

using Complex = std::complex<double>;

void fft(std::vector<Complex>& x, int n, int m) {
    // n = 2^m
    const double pi = 4.0 * std::atan(1.0);

    // Butterfly stages (decimation-in-frequency)
    int l = n;
    for (int k = 0; k < m; ++k) {
        double px = -2.0 * pi / static_cast<double>(l);
        l /= 2;
        for (int j = 0; j < l; ++j) {
            double w = px * static_cast<double>(j);
            Complex twiddle(std::cos(w), std::sin(w));
            for (int i = j; i < n; i += 2 * l) {
                Complex temp = x[i] - x[i + l];
                x[i]        = x[i] + x[i + l];
                x[i + l]    = temp * twiddle;
            }
        }
    }

    // Bit-reversal permutation
    int j = 0;
    for (int i = 0; i < n - 1; ++i) {
        if (i < j) {
            std::swap(x[i], x[j]);
        }
        int half = n / 2;
        while (half <= j) {
            j -= half;
            half /= 2;
        }
        j += half;
    }
}
