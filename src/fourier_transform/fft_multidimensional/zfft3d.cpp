// 3-D Complex FFT Routine (with OpenMP and Cache Blocking)
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

// Inner blocked 3-D FFT computation
void zfft3d_inner(std::vector<Complex>& a, int nx, int ny, int nz,
                  const std::vector<Complex>& wx, const std::vector<Complex>& wy,
                  const std::vector<Complex>& wz,
                  const int lnx[3], const int lny[3], const int lnz[3]) {
    // Phase 1: nz-point FFTs along z-dimension (for each (i,j) pair)
    // #pragma omp parallel for
    for (int j = 0; j < ny; ++j) {
        std::vector<Complex> bz(nz + NP);
        std::vector<Complex> c(nz);

        for (int ii = 0; ii < nx; ii += NBLK) {
            int iend = std::min(ii + NBLK, nx);

            for (int i = ii; i < iend; ++i) {
                // Extract z-pencil
                for (int k = 0; k < nz; ++k) {
                    bz[k] = a[i * ny * nz + j * nz + k];
                }
                fft235(bz, c, wz, nz, lnz);
                // Store back
                for (int k = 0; k < nz; ++k) {
                    a[i * ny * nz + j * nz + k] = bz[k];
                }
            }
        }
    }

    // Phase 2: ny-point FFTs along y-dimension (for each (i,k) pair)
    // #pragma omp parallel for
    for (int k = 0; k < nz; ++k) {
        std::vector<Complex> by(ny + NP);
        std::vector<Complex> c(ny);

        for (int ii = 0; ii < nx; ii += NBLK) {
            int iend = std::min(ii + NBLK, nx);

            for (int i = ii; i < iend; ++i) {
                // Extract y-pencil
                for (int j = 0; j < ny; ++j) {
                    by[j] = a[i * ny * nz + j * nz + k];
                }
                fft235(by, c, wy, ny, lny);
                // Store back
                for (int j = 0; j < ny; ++j) {
                    a[i * ny * nz + j * nz + k] = by[j];
                }
            }
        }

        // Phase 3: nx-point FFTs along x-dimension (column-wise)
        for (int j = 0; j < ny; ++j) {
            std::vector<Complex> col(nx);
            std::vector<Complex> tmp(nx);
            for (int i = 0; i < nx; ++i) {
                col[i] = a[i * ny * nz + j * nz + k];
            }
            fft235(col, tmp, wx, nx, lnx);
            for (int i = 0; i < nx; ++i) {
                a[i * ny * nz + j * nz + k] = col[i];
            }
        }
    }
}

// 3-D complex FFT with OpenMP and cache blocking
// a: input/output array of nx*ny*nz complex values
// nx, ny, nz: transform lengths in x, y, z directions
// iopt: 0 = initialize, -1 = forward, +1 = inverse
void zfft3d(std::vector<Complex>& a, int nx, int ny, int nz, int iopt,
            std::vector<Complex>& wx, std::vector<Complex>& wy,
            std::vector<Complex>& wz) {
    // Initialize twiddle factor tables
    if (iopt == 0) {
        wx.resize(nx);
        wy.resize(ny);
        wz.resize(nz);
        settbl(wx, nx);
        settbl(wy, ny);
        settbl(wz, nz);
        return;
    }

    int ntot = nx * ny * nz;

    // Conjugate for inverse transform
    if (iopt == 1) {
        for (int i = 0; i < ntot; ++i) {
            a[i] = std::conj(a[i]);
        }
    }

    int lnx[3], lny[3], lnz[3];
    factor(nx, lnx);
    factor(ny, lny);
    factor(nz, lnz);

    // Perform blocked 3-D FFT (parallelizable with OpenMP)
    zfft3d_inner(a, nx, ny, nz, wx, wy, wz, lnx, lny, lnz);

    // Scale for inverse transform
    if (iopt == 1) {
        double dn = 1.0 / (static_cast<double>(nx) * static_cast<double>(ny)
                            * static_cast<double>(nz));
        for (int i = 0; i < ntot; ++i) {
            a[i] = std::conj(a[i]) * dn;
        }
    }
}
