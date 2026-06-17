// Two-dimensional FFT based on row-column algorithm

#include <cmath>
#include <complex>
#include <vector>

using Complex = std::complex<double>;

// Forward declaration: assumes a 1-D FFT routine is available
void fft(std::vector<Complex>& col, int n);

void fft2d(std::vector<Complex>& x, int n1, int n2) {
    std::vector<Complex> y(n2 * n1);
    std::vector<Complex> col;

    // Step 1: n2 individual n1-point multicolumn FFTs
    for (int j = 0; j < n2; ++j) {
        col.assign(n1, Complex(0.0, 0.0));
        for (int i = 0; i < n1; ++i) {
            col[i] = x[i * n2 + j];
        }
        fft(col, n1);
        for (int i = 0; i < n1; ++i) {
            x[i * n2 + j] = col[i];
        }
    }

    // Step 2: Transposition (n1 x n2 -> n2 x n1)
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            y[j * n1 + i] = x[i * n2 + j];
        }
    }

    // Step 3: n1 individual n2-point multicolumn FFTs
    for (int i = 0; i < n1; ++i) {
        col.assign(n2, Complex(0.0, 0.0));
        for (int j = 0; j < n2; ++j) {
            col[j] = y[j * n1 + i];
        }
        fft(col, n2);
        for (int j = 0; j < n2; ++j) {
            y[j * n1 + i] = col[j];
        }
    }

    // Step 4: Transposition (n2 x n1 -> n1 x n2)
    for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
            x[i * n2 + j] = y[j * n1 + i];
        }
    }
}
