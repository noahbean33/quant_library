// Six-step FFT algorithm

#include <cmath>
#include <complex>
#include <vector>

using Complex = std::complex<double>;

// Forward declaration: assumes a 1-D FFT routine is available
void fft(std::vector<Complex>& col, int n);

void six_step_fft(std::vector<Complex>& x, std::vector<Complex>& y,
                  int n1, int n2) {
    // x is n1 x n2 (row-major: x[i*n2 + j])
    // y is n2 x n1 (row-major: y[j*n1 + i])
    const double pi = 4.0 * std::atan(1.0);
    const int n = n1 * n2;

    // Precompute twiddle factors: w[i][j] = e^(-2πi * i * j / n)
    std::vector<Complex> w(n);
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            double angle = -2.0 * pi * static_cast<double>(i * j) / static_cast<double>(n);
            w[i * n2 + j] = Complex(std::cos(angle), std::sin(angle));
        }
    }

    // Step 1: Transposition (n1 x n2 -> n2 x n1)
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            y[j * n1 + i] = x[i * n2 + j];
        }
    }

    // Step 2: n1 individual n2-point FFTs on columns of y
    for (int i = 0; i < n1; ++i) {
        std::vector<Complex> col(n2);
        for (int j = 0; j < n2; ++j) col[j] = y[j * n1 + i];
        fft(col, n2);
        for (int j = 0; j < n2; ++j) y[j * n1 + i] = col[j];
    }

    // Steps 3-4: Twiddle factor multiplication + transposition (n2 x n1 -> n1 x n2)
    for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
            x[i * n2 + j] = y[j * n1 + i] * w[i * n2 + j];
        }
    }

    // Step 5: n2 individual n1-point FFTs on columns of x
    for (int j = 0; j < n2; ++j) {
        std::vector<Complex> col(n1);
        for (int i = 0; i < n1; ++i) col[i] = x[i * n2 + j];
        fft(col, n1);
        for (int i = 0; i < n1; ++i) x[i * n2 + j] = col[i];
    }

    // Step 6: Transposition (n1 x n2 -> n2 x n1)
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            y[j * n1 + i] = x[i * n2 + j];
        }
    }
}
