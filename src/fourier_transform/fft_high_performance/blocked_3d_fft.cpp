// Blocked three-dimensional FFT algorithm

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

void blocked_3d_fft(std::vector<Complex>& x, int n1, int n2, int n3, int nb) {
    // x: n1 x n2 x n3 (row-major)

    std::vector<Complex> ywork(static_cast<size_t>(n2) * nb);
    std::vector<Complex> zwork(static_cast<size_t>(n3) * nb);

    // Step 1: n2*n3 individual n1-point FFTs (along dimension 1)
    for (int k = 0; k < n3; ++k) {
        for (int j = 0; j < n2; ++j) {
            std::vector<Complex> col_data(n1);
            for (int i = 0; i < n1; ++i) col_data[i] = x[idx3(i, j, k, n2, n3)];
            fft(col_data, n1);
            for (int i = 0; i < n1; ++i) x[idx3(i, j, k, n2, n3)] = col_data[i];
        }

        // Steps 2-4: Blocked dimension-2 FFTs with transposition
        for (int ii = 0; ii < n1; ii += nb) {
            int ib = std::min(nb, n1 - ii);

            // Step 2: Blocked transposition into ywork
            for (int i = ii; i < ii + ib; ++i) {
                for (int j = 0; j < n2; ++j) {
                    ywork[j * nb + (i - ii)] = x[idx3(i, j, k, n2, n3)];
                }
            }

            // Step 3: ib individual n2-point FFTs
            for (int bi = 0; bi < ib; ++bi) {
                std::vector<Complex> col_data(n2);
                for (int j = 0; j < n2; ++j) col_data[j] = ywork[j * nb + bi];
                fft(col_data, n2);
                for (int j = 0; j < n2; ++j) ywork[j * nb + bi] = col_data[j];
            }

            // Step 4: Blocked transposition back
            for (int j = 0; j < n2; ++j) {
                for (int i = ii; i < ii + ib; ++i) {
                    x[idx3(i, j, k, n2, n3)] = ywork[j * nb + (i - ii)];
                }
            }
        }
    }

    // Steps 5-6: Blocked dimension-3 FFTs with transposition
    for (int j = 0; j < n2; ++j) {
        for (int ii = 0; ii < n1; ii += nb) {
            int ib = std::min(nb, n1 - ii);

            // Blocked transposition into zwork
            for (int i = ii; i < ii + ib; ++i) {
                for (int k = 0; k < n3; ++k) {
                    zwork[k * nb + (i - ii)] = x[idx3(i, j, k, n2, n3)];
                }
            }

            // Step 5: ib individual n3-point FFTs
            for (int bi = 0; bi < ib; ++bi) {
                std::vector<Complex> col_data(n3);
                for (int k = 0; k < n3; ++k) col_data[k] = zwork[k * nb + bi];
                fft(col_data, n3);
                for (int k = 0; k < n3; ++k) zwork[k * nb + bi] = col_data[k];
            }

            // Step 6: Blocked transposition back
            for (int k = 0; k < n3; ++k) {
                for (int i = ii; i < ii + ib; ++i) {
                    x[idx3(i, j, k, n2, n3)] = zwork[k * nb + (i - ii)];
                }
            }
        }
    }
}
