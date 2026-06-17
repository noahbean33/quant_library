// Recursive six-step FFT algorithm

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

using Complex = std::complex<double>;

// Forward declaration: a standard in-cache FFT
void fft(std::vector<Complex>& a, std::vector<Complex>& b, int n);

// Decompose n into two factors n1, n2 close to sqrt(n)
static void get_n1_n2(int n, int& n1, int& n2) {
    n1 = static_cast<int>(std::sqrt(static_cast<double>(n)));
    while (n % n1 != 0) --n1;
    n2 = n / n1;
}

static constexpr int CACHE_SIZE = 1024;  // base-case threshold (tunable)
static constexpr int NB = 16;            // block size for transpositions

void recursive_fft(std::vector<Complex>& a, std::vector<Complex>& b,
                   const std::vector<Complex>& w, int n) {
    if (n <= CACHE_SIZE) {
        fft(a, b, n);
        return;
    }

    // Step 0: Decompose n into n1 x n2
    int n1, n2;
    get_n1_n2(n, n1, n2);

    // Step 1: Blocked transposition (column-major to row-major of n2 x n1)
    for (int ii = 0; ii < n1; ii += NB) {
        int ib = std::min(NB, n1 - ii);
        for (int jj = 0; jj < n2; jj += NB) {
            int jb = std::min(NB, n2 - jj);
            for (int i = ii; i < ii + ib; ++i) {
                for (int j = jj; j < jj + jb; ++j) {
                    b[j + i * n2] = a[i + j * n1];
                }
            }
        }
    }

    // Step 2: n1 individual n2-point FFTs (recursive)
    for (int i = 0; i < n1; ++i) {
        std::vector<Complex> sub(b.begin() + i * n2, b.begin() + (i + 1) * n2);
        std::vector<Complex> tmp(n2);
        recursive_fft(sub, tmp, w, n2);
        std::copy(sub.begin(), sub.end(), a.begin() + i * n2);
    }

    // Step 3: Twiddle factor multiplication
    for (int i = 0; i < n; ++i) {
        a[i] *= w[i];
    }

    // Step 4: Blocked transposition
    for (int jj = 0; jj < n2; jj += NB) {
        int jb = std::min(NB, n2 - jj);
        for (int ii = 0; ii < n1; ii += NB) {
            int ib = std::min(NB, n1 - ii);
            for (int j = jj; j < jj + jb; ++j) {
                for (int i = ii; i < ii + ib; ++i) {
                    b[i + j * n1] = a[j + i * n2];
                }
            }
        }
    }

    // Step 5: n2 individual n1-point FFTs (recursive)
    for (int j = 0; j < n2; ++j) {
        std::vector<Complex> sub(b.begin() + j * n1, b.begin() + (j + 1) * n1);
        std::vector<Complex> tmp(n1);
        recursive_fft(sub, tmp, w, n1);
        std::copy(sub.begin(), sub.end(), a.begin() + j * n1);
    }

    // Step 6: Blocked transposition
    for (int ii = 0; ii < n1; ii += NB) {
        int ib = std::min(NB, n1 - ii);
        for (int jj = 0; jj < n2; jj += NB) {
            int jb = std::min(NB, n2 - jj);
            for (int i = ii; i < ii + ib; ++i) {
                for (int j = jj; j < jj + jb; ++j) {
                    b[j + i * n2] = a[i + j * n1];
                }
            }
        }
    }
}
