// Parallelization of six-step FFT with OpenMP loop collapse

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

using Complex = std::complex<double>;

// Forward declaration: assumes a 1-D FFT routine is available
void fft(std::vector<Complex>& col, int n);

void openmp_six_step_fft(std::vector<Complex>& x, std::vector<Complex>& y,
                         const std::vector<Complex>& w,
                         int n1, int n2, int nb) {
    // x: n1 x n2 row-major
    // y: n2 x n1 row-major
    // w: n1 x n2 twiddle factors (row-major)

    #pragma omp parallel
    {
        // Step 1: Blocked transposition (n1 x n2 -> n2 x n1)
        #pragma omp for collapse(2) schedule(static)
        for (int ii = 0; ii < n1; ii += nb) {
            for (int jj = 0; jj < n2; jj += nb) {
                int ib = std::min(nb, n1 - ii);
                int jb = std::min(nb, n2 - jj);
                for (int i = ii; i < ii + ib; ++i) {
                    for (int j = jj; j < jj + jb; ++j) {
                        y[j * n1 + i] = x[i * n2 + j];
                    }
                }
            }
        }

        // Step 2: n1 individual n2-point column FFTs
        #pragma omp for schedule(static)
        for (int i = 0; i < n1; ++i) {
            std::vector<Complex> col(n2);
            for (int j = 0; j < n2; ++j) col[j] = y[j * n1 + i];
            fft(col, n2);
            for (int j = 0; j < n2; ++j) y[j * n1 + i] = col[j];
        }

        // Steps 3-4: Twiddle factor multiplication + blocked transposition
        #pragma omp for collapse(2) schedule(static)
        for (int jj = 0; jj < n2; jj += nb) {
            for (int ii = 0; ii < n1; ii += nb) {
                int jb = std::min(nb, n2 - jj);
                int ib = std::min(nb, n1 - ii);
                for (int j = jj; j < jj + jb; ++j) {
                    for (int i = ii; i < ii + ib; ++i) {
                        x[i * n2 + j] = y[j * n1 + i] * w[i * n2 + j];
                    }
                }
            }
        }

        // Step 5: n2 individual n1-point column FFTs
        #pragma omp for schedule(static)
        for (int j = 0; j < n2; ++j) {
            std::vector<Complex> col(n1);
            for (int i = 0; i < n1; ++i) col[i] = x[i * n2 + j];
            fft(col, n1);
            for (int i = 0; i < n1; ++i) x[i * n2 + j] = col[i];
        }

        // Step 6: Blocked transposition (n1 x n2 -> n2 x n1)
        #pragma omp for collapse(2) schedule(static)
        for (int ii = 0; ii < n1; ii += nb) {
            for (int jj = 0; jj < n2; jj += nb) {
                int ib = std::min(nb, n1 - ii);
                int jb = std::min(nb, n2 - jj);
                for (int i = ii; i < ii + ib; ++i) {
                    for (int j = jj; j < jj + jb; ++j) {
                        y[j * n1 + i] = x[i * n2 + j];
                    }
                }
            }
        }
    }
}
