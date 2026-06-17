// Blocked six-step FFT algorithm

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

using Complex = std::complex<double>;

// Forward declaration: assumes a 1-D FFT routine is available
void fft(std::vector<Complex>& col, int n);

void blocked_six_step_fft(std::vector<Complex>& x, std::vector<Complex>& y,
                          const std::vector<Complex>& w,
                          int n1, int n2, int nb) {
    // x: n1 x n2 row-major, y: n2 x n1 row-major
    // w: n1 x n2 twiddle factors (row-major)
    // nb: cache block size

    std::vector<Complex> work(static_cast<size_t>(n2) * nb);

    for (int ii = 0; ii < n1; ii += nb) {
        int ib = std::min(nb, n1 - ii);

        // Step 1: Blocked transposition into work buffer
        for (int jj = 0; jj < n2; jj += nb) {
            int jb = std::min(nb, n2 - jj);
            for (int i = ii; i < ii + ib; ++i) {
                for (int j = jj; j < jj + jb; ++j) {
                    work[j * nb + (i - ii)] = x[i * n2 + j];
                }
            }
        }

        // Step 2: ib individual n2-point FFTs
        for (int i = 0; i < ib; ++i) {
            std::vector<Complex> col(n2);
            for (int j = 0; j < n2; ++j) col[j] = work[j * nb + i];
            fft(col, n2);
            for (int j = 0; j < n2; ++j) work[j * nb + i] = col[j];
        }

        // Steps 3-4: Twiddle factor multiplication + transposition back
        for (int j = 0; j < n2; ++j) {
            for (int i = ii; i < ii + ib; ++i) {
                x[i * n2 + j] = work[j * nb + (i - ii)] * w[i * n2 + j];
            }
        }
    }

    for (int jj = 0; jj < n2; jj += nb) {
        int jb = std::min(nb, n2 - jj);

        // Step 5: jb individual n1-point FFTs
        for (int j = jj; j < jj + jb; ++j) {
            std::vector<Complex> col(n1);
            for (int i = 0; i < n1; ++i) col[i] = x[i * n2 + j];
            fft(col, n1);
            for (int i = 0; i < n1; ++i) x[i * n2 + j] = col[i];
        }

        // Step 6: Blocked transposition
        for (int i = 0; i < n1; ++i) {
            for (int j = jj; j < jj + jb; ++j) {
                y[j * n1 + i] = x[i * n2 + j];
            }
        }
    }
}
