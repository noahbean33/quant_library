// Blocked nine-step FFT algorithm (3-D)

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

using Complex = std::complex<double>;

// Forward declaration: assumes a 1-D FFT routine is available
void fft(std::vector<Complex>& col, int n);

// Helper: 3-D index in row-major order (n1 x n2 x n3)
static inline int idx3(int i, int j, int k, int n2, int n3) {
    return (i * n2 + j) * n3 + k;
}

void blocked_nine_step_fft(std::vector<Complex>& x, std::vector<Complex>& y,
                           const std::vector<Complex>& u2,
                           const std::vector<Complex>& u3,
                           int n1, int n2, int n3, int nb) {
    // x: n1 x n2 x n3 (row-major)
    // y: n3 x n2 x n1 (row-major output)
    // u2: n3 x n2 twiddle factors
    // u3: n1 x n2 x n3 twiddle factors

    std::vector<Complex> ywork(static_cast<size_t>(n2) * nb);
    std::vector<Complex> zwork(static_cast<size_t>(n3) * nb);

    // --- Phase 1: FFTs along dimension 3 ---
    for (int j = 0; j < n2; ++j) {
        for (int ii = 0; ii < n1; ii += nb) {
            int ib = std::min(nb, n1 - ii);

            // Step 1: Blocked transposition into zwork
            for (int kk = 0; kk < n3; kk += nb) {
                int kb = std::min(nb, n3 - kk);
                for (int i = ii; i < ii + ib; ++i) {
                    for (int k = kk; k < kk + kb; ++k) {
                        zwork[k * nb + (i - ii)] = x[idx3(i, j, k, n2, n3)];
                    }
                }
            }

            // Step 2: ib individual n3-point FFTs
            for (int bi = 0; bi < ib; ++bi) {
                std::vector<Complex> col(n3);
                for (int k = 0; k < n3; ++k) col[k] = zwork[k * nb + bi];
                fft(col, n3);
                for (int k = 0; k < n3; ++k) zwork[k * nb + bi] = col[k];
            }

            // Steps 3-4: Twiddle multiplication + write back
            for (int k = 0; k < n3; ++k) {
                for (int i = ii; i < ii + ib; ++i) {
                    x[idx3(i, j, k, n2, n3)] = zwork[k * nb + (i - ii)] * u2[k * n2 + j];
                }
            }
        }
    }

    // --- Phase 2: FFTs along dimension 2 ---
    for (int k = 0; k < n3; ++k) {
        for (int ii = 0; ii < n1; ii += nb) {
            int ib = std::min(nb, n1 - ii);

            // Blocked transposition into ywork
            for (int jj = 0; jj < n2; jj += nb) {
                int jb = std::min(nb, n2 - jj);
                for (int i = ii; i < ii + ib; ++i) {
                    for (int j = jj; j < jj + jb; ++j) {
                        ywork[j * nb + (i - ii)] = x[idx3(i, j, k, n2, n3)];
                    }
                }
            }

            // Step 5: ib individual n2-point FFTs
            for (int bi = 0; bi < ib; ++bi) {
                std::vector<Complex> col(n2);
                for (int j = 0; j < n2; ++j) col[j] = ywork[j * nb + bi];
                fft(col, n2);
                for (int j = 0; j < n2; ++j) ywork[j * nb + bi] = col[j];
            }

            // Steps 6-7: Twiddle multiplication + write back
            for (int j = 0; j < n2; ++j) {
                for (int i = ii; i < ii + ib; ++i) {
                    x[idx3(i, j, k, n2, n3)] =
                        ywork[j * nb + (i - ii)] * u3[idx3(i, j, k, n2, n3)];
                }
            }
        }

        // Step 8: n2 individual n1-point FFTs along dimension 1
        for (int j = 0; j < n2; ++j) {
            std::vector<Complex> col(n1);
            for (int i = 0; i < n1; ++i) col[i] = x[idx3(i, j, k, n2, n3)];
            fft(col, n1);
            for (int i = 0; i < n1; ++i) x[idx3(i, j, k, n2, n3)] = col[i];
        }
    }

    // Step 9: Blocked transposition (n1 x n2 x n3 -> n3 x n2 x n1)
    for (int ii = 0; ii < n1; ii += nb) {
        int ib = std::min(nb, n1 - ii);
        for (int jj = 0; jj < n2; jj += nb) {
            int jb = std::min(nb, n2 - jj);
            for (int kk = 0; kk < n3; kk += nb) {
                int kb = std::min(nb, n3 - kk);
                for (int i = ii; i < ii + ib; ++i) {
                    for (int j = jj; j < jj + jb; ++j) {
                        for (int k = kk; k < kk + kb; ++k) {
                            y[(k * n2 + j) * n1 + i] = x[idx3(i, j, k, n2, n3)];
                        }
                    }
                }
            }
        }
    }
}
