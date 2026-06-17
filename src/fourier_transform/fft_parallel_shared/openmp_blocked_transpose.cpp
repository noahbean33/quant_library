// OpenMP cache-blocked matrix transposition

#include <complex>
#include <vector>
#include <algorithm>

using Complex = std::complex<double>;

void openmp_blocked_transpose(const std::vector<Complex>& x,
                              std::vector<Complex>& y,
                              int n1, int n2, int nb) {
    // x: n1 x n2 row-major  (x[i*n2 + j])
    // y: n2 x n1 row-major  (y[j*n1 + i])

    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n1; ii += nb) {
        int ib = std::min(nb, n1 - ii);
        for (int jj = 0; jj < n2; jj += nb) {
            int jb = std::min(nb, n2 - jj);
            for (int i = ii; i < ii + ib; ++i) {
                for (int j = jj; j < jj + jb; ++j) {
                    y[j * n1 + i] = x[i * n2 + j];
                }
            }
        }
    }
}
