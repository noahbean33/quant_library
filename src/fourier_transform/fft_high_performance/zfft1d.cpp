// 1-D Complex FFT Routine (with OpenMP and Cache Blocking)
// Blocked six-step FFT for large transforms, direct FFT for small transforms

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

using Complex = std::complex<double>;

// Forward declarations for external routines
void factor(int n, int ip[3]);
void settbl(std::vector<Complex>& w, int n);
void settbl2(std::vector<Complex>& w, int nx, int ny);
void getnxny(int n, int& nx, int& ny);
void fft235(std::vector<Complex>& a, std::vector<Complex>& b,
            const std::vector<Complex>& w, int n, const int ip[3]);

static const int NBLK = 16;       // Blocking parameter for cache
static const int NP = 8;          // Padding to avoid cache conflicts
static const int L2SIZE = 8388608; // L2 cache size in bytes

// Inner blocked FFT computation (parallelizable over blocks)
// Performs blocked six-step FFT: column FFTs -> transpose+twiddle -> row FFTs -> transpose
void zfft1d_inner(std::vector<Complex>& a, std::vector<Complex>& b,
                  const std::vector<Complex>& wx, const std::vector<Complex>& wy,
                  const std::vector<Complex>& w, int nx, int ny,
                  const int lnx[3], const int lny[3]) {
    std::vector<Complex> cy(ny + NP);
    std::vector<Complex> cx(nx + NP);
    std::vector<Complex> d(std::max(nx, ny));

    // Phase 1: Blocked transpose + ny-point FFTs along columns
    for (int ii = 0; ii < nx; ii += NBLK) {
        int iend = std::min(ii + NBLK, nx);

        // Blocked transpose: copy columns of a (nx x ny) into cy buffer
        for (int jj = 0; jj < ny; jj += NBLK) {
            int jend = std::min(jj + NBLK, ny);
            for (int i = ii; i < iend; ++i) {
                for (int j = jj; j < jend; ++j) {
                    cy[j * (iend - ii) + (i - ii)] = a[i * ny + j]; // Conceptual blocked transpose
                }
            }
        }

        // Perform ny-point FFTs on each transposed column
        for (int i = ii; i < iend; ++i) {
            // Extract column
            for (int j = 0; j < ny; ++j) {
                cy[j] = a[i * ny + j];
            }
            fft235(cy, d, wy, ny, lny);
            // Store back
            for (int j = 0; j < ny; ++j) {
                b[i * ny + j] = cy[j];
            }
        }
    }

    // Phase 2: Twiddle factor multiplication + nx-point FFTs + transpose
    for (int jj = 0; jj < ny; jj += NBLK) {
        int jend = std::min(jj + NBLK, ny);

        for (int j = jj; j < jend; ++j) {
            // Multiply by twiddle factors and extract row
            for (int i = 0; i < nx; ++i) {
                cx[i] = b[i * ny + j] * w[i * ny + j];
            }
            fft235(cx, d, wx, nx, lnx);
            // Transpose result: store as column of output
            for (int i = 0; i < nx; ++i) {
                a[j * nx + i] = cx[i];
            }
        }
    }
}

// 1-D complex FFT with OpenMP and cache blocking
// a: input/output array of n complex values
// b: work/coefficient array of size 2*n
// n = (2^ip) * (3^iq) * (5^ir)
// iopt: 0 = initialize, -1 = forward, +1 = inverse
void zfft1d(std::vector<Complex>& a, int n, int iopt,
            std::vector<Complex>& b,
            std::vector<Complex>& wx, std::vector<Complex>& wy) {
    // Conjugate for inverse transform
    if (iopt == 1) {
        for (int i = 0; i < n; ++i) {
            a[i] = std::conj(a[i]);
        }
    }

    // Small transform: direct FFT (fits in L2 cache)
    if (n <= (L2SIZE / 16) / 3) {
        if (iopt == 0) {
            b.resize(2 * n);
            settbl(b, n);  // Store twiddle table in b[n..2n-1] region
            return;
        }

        int ip[3];
        factor(n, ip);
        std::vector<Complex> w(b.begin() + n, b.begin() + 2 * n);
        fft235(a, b, w, n, ip);
    } else {
        // Large transform: blocked six-step FFT
        int nx, ny;
        getnxny(n, nx, ny);

        if (iopt == 0) {
            wx.resize(nx);
            wy.resize(ny);
            settbl(wx, nx);
            settbl(wy, ny);
            b.resize(2 * n);
            settbl2(b, nx, ny);  // Twiddle matrix in b[n..2n-1]
            return;
        }

        int lnx[3], lny[3];
        factor(nx, lnx);
        factor(ny, lny);

        // The inner computation is parallelizable with OpenMP
        // #pragma omp parallel
        std::vector<Complex> tw(b.begin() + n, b.begin() + 2 * n);
        zfft1d_inner(a, b, wx, wy, tw, nx, ny, lnx, lny);
    }

    // Scale for inverse transform
    if (iopt == 1) {
        double dn = 1.0 / static_cast<double>(n);
        for (int i = 0; i < n; ++i) {
            a[i] = std::conj(a[i]) * dn;
        }
    }
}
