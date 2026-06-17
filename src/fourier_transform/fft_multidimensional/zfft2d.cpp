// 2-D Complex FFT Routine (with OpenMP and Cache Blocking)
// Blocked row-column approach for cache efficiency

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

using Complex = std::complex<double>;

// Forward declarations for external routines
void factor(int n, int ip[3]);
void settbl(std::vector<Complex>& w, int n);
void fft235(std::vector<Complex>& a, std::vector<Complex>& b,
            const std::vector<Complex>& w, int n, const int ip[3]);

static const int NBLK = 16;
static const int NP = 8;

// Inner blocked 2-D FFT computation
void zfft2d_inner(std::vector<Complex>& a, int nx, int ny,
                  const std::vector<Complex>& wx, const std::vector<Complex>& wy,
                  const int lnx[3], const int lny[3]) {
    std::vector<Complex> b(ny + NP);
    std::vector<Complex> c(std::max(nx, ny));

    // Phase 1: Blocked transpose + ny-point FFTs along y-dimension
    // Process NBLK columns at a time for cache efficiency
    for (int ii = 0; ii < nx; ii += NBLK) {
        int iend = std::min(ii + NBLK, nx);

        // Blocked transpose into local buffer and perform FFTs
        for (int i = ii; i < iend; ++i) {
            for (int j = 0; j < ny; ++j) {
                b[j] = a[i * ny + j];
            }
            fft235(b, c, wy, ny, lny);
            for (int j = 0; j < ny; ++j) {
                a[i * ny + j] = b[j];
            }
        }
    }

    // Phase 2: nx-point FFTs along x-dimension (column-wise)
    // #pragma omp parallel for
    for (int j = 0; j < ny; ++j) {
        std::vector<Complex> col(nx);
        std::vector<Complex> tmp(nx);
        for (int i = 0; i < nx; ++i) {
            col[i] = a[i * ny + j];
        }
        fft235(col, tmp, wx, nx, lnx);
        for (int i = 0; i < nx; ++i) {
            a[i * ny + j] = col[i];
        }
    }
}

// 2-D complex FFT with OpenMP and cache blocking
// a: input/output array of nx*ny complex values (row-major)
// nx, ny: transform lengths in x and y directions
// iopt: 0 = initialize, -1 = forward, +1 = inverse
void zfft2d(std::vector<Complex>& a, int nx, int ny, int iopt,
            std::vector<Complex>& wx, std::vector<Complex>& wy) {
    // Initialize twiddle factor tables
    if (iopt == 0) {
        wx.resize(nx);
        wy.resize(ny);
        settbl(wx, nx);
        settbl(wy, ny);
        return;
    }

    // Conjugate for inverse transform
    if (iopt == 1) {
        for (int i = 0; i < nx * ny; ++i) {
            a[i] = std::conj(a[i]);
        }
    }

    int lnx[3], lny[3];
    factor(nx, lnx);
    factor(ny, lny);

    // Perform blocked 2-D FFT (parallelizable with OpenMP)
    zfft2d_inner(a, nx, ny, wx, wy, lnx, lny);

    // Scale for inverse transform
    if (iopt == 1) {
        double dn = 1.0 / (static_cast<double>(nx) * static_cast<double>(ny));
        for (int i = 0; i < nx * ny; ++i) {
            a[i] = std::conj(a[i]) * dn;
        }
    }
}
