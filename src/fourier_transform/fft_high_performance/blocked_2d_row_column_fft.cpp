// Blocked two-dimensional FFT based on row-column algorithm

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

using Complex = std::complex<double>;

// Forward declaration: assumes a 1-D FFT routine is available
void fft(std::vector<Complex>& col, int n);

void blocked_2d_row_column_fft(std::vector<Complex>& x, int n1, int n2, int nb) {
    // x: n1 x n2 (row-major: x[i*n2 + j])

    // Step 1: n2 individual n1-point column FFTs
    for (int j = 0; j < n2; ++j) {
        std::vector<Complex> col_data(n1);
        for (int i = 0; i < n1; ++i) col_data[i] = x[i * n2 + j];
        fft(col_data, n1);
        for (int i = 0; i < n1; ++i) x[i * n2 + j] = col_data[i];
    }

    std::vector<Complex> work(static_cast<size_t>(n2) * nb);

    for (int ii = 0; ii < n1; ii += nb) {
        int ib = std::min(nb, n1 - ii);

        // Step 2: Blocked transposition into work buffer
        for (int jj = 0; jj < n2; jj += nb) {
            int jb = std::min(nb, n2 - jj);
            for (int i = ii; i < ii + ib; ++i) {
                for (int j = jj; j < jj + jb; ++j) {
                    work[j * nb + (i - ii)] = x[i * n2 + j];
                }
            }
        }

        // Step 3: ib individual n2-point row FFTs
        for (int bi = 0; bi < ib; ++bi) {
            std::vector<Complex> row(n2);
            for (int j = 0; j < n2; ++j) row[j] = work[j * nb + bi];
            fft(row, n2);
            for (int j = 0; j < n2; ++j) work[j * nb + bi] = row[j];
        }

        // Step 4: Blocked transposition back
        for (int j = 0; j < n2; ++j) {
            for (int i = ii; i < ii + ib; ++i) {
                x[i * n2 + j] = work[j * nb + (i - ii)];
            }
        }
    }
}
